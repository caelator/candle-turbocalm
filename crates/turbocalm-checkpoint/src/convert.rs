use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use std::path::{Path, PathBuf};
use tracing::{error, info, warn};
use turbocalm_core::{auto_device, CALMConfig, AutoencoderConfig};

use crate::{
    download::{CheckpointDownloader, CALMCheckpoint},
    parser::StateDictParser,
    remapping::{RemappingPresets, TensorNameRemapper},
    verification::ShapeVerifier,
    manifest::{CALMModelManifest, ManifestManager},
};

/// Convert command arguments
#[derive(Debug, Args)]
pub struct ConvertArgs {
    #[command(subcommand)]
    pub command: ConvertCommand,
}

/// Convert subcommands
#[derive(Debug, Subcommand)]
pub enum ConvertCommand {
    /// Download and convert a model from HuggingFace Hub
    FromHub(FromHubArgs),
    /// Convert a local model to CALM format
    FromLocal(FromLocalArgs),
    /// List converted models
    List(ListArgs),
    /// Delete a converted model
    Delete(DeleteArgs),
}

/// Convert from HuggingFace Hub
#[derive(Debug, Args)]
pub struct FromHubArgs {
    /// HuggingFace model ID (e.g., "meta-llama/Meta-Llama-3-8B")
    #[arg(value_name = "MODEL_ID")]
    pub model_id: String,

    /// Output directory for converted model
    #[arg(short, long, value_name = "DIR")]
    pub output: Option<PathBuf>,

    /// Force overwrite if model already exists
    #[arg(short, long)]
    pub force: bool,

    /// Skip shape verification
    #[arg(long)]
    pub skip_verification: bool,

    /// Use strict verification mode
    #[arg(long)]
    pub strict_verification: bool,

    /// Manifest directory
    #[arg(long, value_name = "DIR")]
    pub manifest_dir: Option<PathBuf>,

    /// Custom CALM configuration file
    #[arg(long, value_name = "FILE")]
    pub calm_config: Option<PathBuf>,

    /// Custom autoencoder configuration file
    #[arg(long, value_name = "FILE")]
    pub autoencoder_config: Option<PathBuf>,

    /// Additional tags for the model
    #[arg(long, value_delimiter = ',')]
    pub tags: Vec<String>,
}

/// Convert from local files
#[derive(Debug, Args)]
pub struct FromLocalArgs {
    /// Input model files (safetensors or .bin)
    #[arg(value_name = "FILES", required = true)]
    pub input_files: Vec<PathBuf>,

    /// Model configuration file
    #[arg(short, long, value_name = "FILE")]
    pub config: PathBuf,

    /// Output directory for converted model
    #[arg(short, long, value_name = "DIR")]
    pub output: PathBuf,

    /// Model name for the conversion
    #[arg(short, long)]
    pub name: String,

    /// Skip shape verification
    #[arg(long)]
    pub skip_verification: bool,

    /// Use strict verification mode
    #[arg(long)]
    pub strict_verification: bool,

    /// Remapping preset to use
    #[arg(long, default_value = "huggingface-llama-to-calm")]
    pub remapping: String,
}

/// List converted models
#[derive(Debug, Args)]
pub struct ListArgs {
    /// Manifest directory
    #[arg(long, value_name = "DIR")]
    pub manifest_dir: Option<PathBuf>,

    /// Show detailed information
    #[arg(short, long)]
    pub verbose: bool,
}

/// Delete a converted model
#[derive(Debug, Args)]
pub struct DeleteArgs {
    /// Model name to delete
    #[arg(value_name = "MODEL_NAME")]
    pub model_name: String,

    /// Also delete the model files
    #[arg(long)]
    pub delete_files: bool,

    /// Manifest directory
    #[arg(long, value_name = "DIR")]
    pub manifest_dir: Option<PathBuf>,

    /// Force deletion without confirmation
    #[arg(short, long)]
    pub force: bool,
}

/// Main convert command handler
pub struct ConvertHandler {
    pub manifest_manager: ManifestManager,
    device: candle_core::Device,
}

impl ConvertHandler {
    /// Create a new convert handler
    pub fn new(manifest_dir: Option<PathBuf>) -> Result<Self> {
        let manifest_dir = manifest_dir.unwrap_or_else(|| {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".turbocalm")
                .join("manifests")
        });

        let manifest_manager = ManifestManager::new(manifest_dir)?;
        let device = auto_device()?;

        Ok(Self {
            manifest_manager,
            device,
        })
    }

    /// Handle convert command
    pub fn handle_command(&self, command: ConvertCommand) -> Result<()> {
        match command {
            ConvertCommand::FromHub(args) => self.convert_from_hub(args),
            ConvertCommand::FromLocal(args) => self.convert_from_local(args),
            ConvertCommand::List(args) => self.list_models(args),
            ConvertCommand::Delete(args) => self.delete_model(args),
        }
    }

    /// Convert from HuggingFace Hub
    fn convert_from_hub(&self, args: FromHubArgs) -> Result<()> {
        info!("Converting model from HuggingFace Hub: {}", args.model_id);

        // Download checkpoint
        let downloader = CheckpointDownloader::new()?;
        let checkpoint = downloader.download_calm_checkpoint(&args.model_id)
            .with_context(|| format!("Failed to download model: {}", args.model_id))?;

        // Determine output directory
        let output_dir = args.output.unwrap_or_else(|| {
            let model_name = args.model_id.split('/').last().unwrap_or(&args.model_id);
            PathBuf::from(format!("./converted_models/{}", model_name))
        });

        // Load custom configs if provided
        let calm_config = if let Some(config_path) = args.calm_config {
            CALMConfig::from_json_file(config_path.to_str().unwrap())?
        } else {
            checkpoint.calm_config()
        };

        let autoencoder_config = if let Some(config_path) = args.autoencoder_config {
            Some(AutoencoderConfig::from_json_file(config_path.to_str().unwrap())?)
        } else {
            Some(checkpoint.autoencoder_config())
        };

        // Parse model tensors
        let parser = StateDictParser::new(self.device.clone());
        let tensors = parser.parse_model_files(checkpoint.model_paths())
            .with_context(|| "Failed to parse model tensors")?;

        // Apply tensor name remapping
        let remapper = RemappingPresets::huggingface_llama_to_calm();
        let remapped_tensors = remapper.remap_tensors(tensors)
            .with_context(|| "Failed to remap tensor names")?;

        // Verify shapes if requested
        let verification_report = if !args.skip_verification {
            let verifier = ShapeVerifier::for_calm_model(&calm_config, args.strict_verification);
            Some(verifier.verify_model_shapes(&remapped_tensors)?)
        } else {
            None
        };

        // Save converted model
        self.save_converted_model(
            &args.model_id,
            &output_dir,
            &calm_config,
            autoencoder_config.as_ref(),
            &remapped_tensors,
            verification_report.as_ref(),
            args.tags,
        )?;

        info!("Model conversion completed successfully");
        info!("Output directory: {}", output_dir.display());

        Ok(())
    }

    /// Convert from local files
    fn convert_from_local(&self, args: FromLocalArgs) -> Result<()> {
        info!("Converting model from local files");

        // Load configuration
        let config_content = std::fs::read_to_string(&args.config)
            .with_context(|| format!("Failed to read config file: {}", args.config.display()))?;

        let calm_config: CALMConfig = serde_json::from_str(&config_content)
            .with_context(|| "Failed to parse CALM config")?;

        // Parse model tensors
        let parser = StateDictParser::new(self.device.clone());
        let tensors = parser.parse_model_files(&args.input_files)
            .with_context(|| "Failed to parse model tensors")?;

        // Apply tensor name remapping
        let remapper = match args.remapping.as_str() {
            "huggingface-llama-to-calm" => RemappingPresets::huggingface_llama_to_calm(),
            "calm-to-huggingface-llama" => RemappingPresets::calm_to_huggingface_llama(),
            "identity" => RemappingPresets::identity(),
            _ => {
                warn!("Unknown remapping preset: {}, using identity", args.remapping);
                RemappingPresets::identity()
            }
        };

        let remapped_tensors = remapper.remap_tensors(tensors)
            .with_context(|| "Failed to remap tensor names")?;

        // Verify shapes if requested
        let verification_report = if !args.skip_verification {
            let verifier = ShapeVerifier::for_calm_model(&calm_config, args.strict_verification);
            Some(verifier.verify_model_shapes(&remapped_tensors)?)
        } else {
            None
        };

        // Save converted model
        self.save_converted_model(
            &args.name,
            &args.output,
            &calm_config,
            None,
            &remapped_tensors,
            verification_report.as_ref(),
            Vec::new(),
        )?;

        info!("Model conversion completed successfully");
        info!("Output directory: {}", args.output.display());

        Ok(())
    }

    /// List converted models
    fn list_models(&self, args: ListArgs) -> Result<()> {
        info!("Listing converted models");

        let summaries = self.manifest_manager.list_manifests()
            .with_context(|| "Failed to list manifests")?;

        if summaries.is_empty() {
            info!("No converted models found");
            return Ok(());
        }

        info!("Found {} converted models:", summaries.len());
        for summary in summaries {
            if args.verbose {
                // Load full manifest for detailed info
                if let Ok(manifest) = self.manifest_manager.load_manifest(&summary.model_name) {
                    manifest.display_summary();
                    println!(); // Add spacing
                } else {
                    summary.display();
                }
            } else {
                summary.display();
            }
        }

        Ok(())
    }

    /// Delete a converted model
    fn delete_model(&self, args: DeleteArgs) -> Result<()> {
        info!("Deleting model: {}", args.model_name);

        // Confirm deletion if not forced
        if !args.force {
            print!("Are you sure you want to delete '{}'? ", args.model_name);
            if args.delete_files {
                print!("This will also delete all model files. ");
            }
            println!("(y/N)");

            let mut input = String::new();
            std::io::stdin().read_line(&mut input)?;

            if input.trim().to_lowercase() != "y" && input.trim().to_lowercase() != "yes" {
                info!("Deletion cancelled");
                return Ok(());
            }
        }

        self.manifest_manager.delete_manifest(&args.model_name, args.delete_files)
            .with_context(|| format!("Failed to delete model: {}", args.model_name))?;

        info!("Successfully deleted model: {}", args.model_name);
        Ok(())
    }

    /// Save converted model to disk
    fn save_converted_model(
        &self,
        model_id: &str,
        output_dir: &Path,
        calm_config: &CALMConfig,
        autoencoder_config: Option<&AutoencoderConfig>,
        tensors: &std::collections::HashMap<String, candle_core::Tensor>,
        verification_report: Option<&crate::verification::VerificationReport>,
        tags: Vec<String>,
    ) -> Result<()> {
        info!("Saving converted model to: {}", output_dir.display());

        // Create output directory
        std::fs::create_dir_all(output_dir)?;

        // Save tensors as safetensors
        let safetensors_path = output_dir.join("model.safetensors");
        self.save_tensors_as_safetensors(tensors, &safetensors_path)?;

        // Save configurations
        let config_path = output_dir.join("config.json");
        calm_config.to_json_file(config_path.to_str().unwrap())?;

        let ae_config_path = if let Some(ae_config) = autoencoder_config {
            let path = output_dir.join("autoencoder_config.json");
            ae_config.to_json_file(path.to_str().unwrap())?;
            Some(path)
        } else {
            None
        };

        // Create manifest
        let mut manifest = CALMModelManifest::new(
            model_id,
            calm_config.clone(),
            autoencoder_config.cloned(),
        );

        // Add files to manifest
        manifest.add_safetensors_file("model.safetensors", &safetensors_path, None);
        manifest.add_config_file("config.json", &config_path);

        if let Some(ae_path) = ae_config_path {
            manifest.add_config_file("autoencoder_config.json", &ae_path);
        }

        // Add verification results if available
        if let Some(verification) = verification_report {
            let verification_summary = crate::manifest::VerificationSummary::from_verification_report(verification);
            manifest.set_verification_results(verification_summary);

            if verification.passed() {
                info!("Shape verification passed");
            } else {
                warn!("Shape verification failed with {} errors", verification.error_count);
            }
        }

        // Add tags
        manifest.metadata.tags = tags;

        // Save manifest
        self.manifest_manager.save_manifest(&manifest)?;

        info!("Model saved successfully");
        Ok(())
    }

    /// Save tensors as safetensors file
    pub fn save_tensors_as_safetensors(
        &self,
        tensors: &std::collections::HashMap<String, candle_core::Tensor>,
        output_path: &Path,
    ) -> Result<()> {
        info!("Saving {} tensors to safetensors format", tensors.len());

        // Convert tensors to safetensors format
        let mut safetensor_data = std::collections::HashMap::new();

        for (name, tensor) in tensors {
            // Convert tensor to bytes
            let tensor_data = tensor.to_dtype(candle_core::DType::F32)?;
            let raw_data = tensor_data.flatten_all()?.to_vec1::<f32>()?;
            let bytes: Vec<u8> = raw_data.iter()
                .flat_map(|&f| f.to_le_bytes())
                .collect();

            // Get shape and dtype info
            let shape: Vec<usize> = tensor.shape().dims().to_vec();

            safetensor_data.insert(
                name.clone(),
                (safetensors::Dtype::F32, shape, bytes),
            );
        }

        // Serialize to safetensors format using tensor_tools
        let st_data: std::collections::HashMap<String, safetensors::tensor::TensorView<'_>> = safetensor_data
            .iter()
            .map(|(name, (dtype, shape, data))| {
                (name.clone(), safetensors::tensor::TensorView::new(*dtype, shape.clone(), data).unwrap())
            })
            .collect();
        safetensors::serialize_to_file(&st_data, &None, output_path)
            .with_context(|| format!("Failed to write safetensors file: {}", output_path.display()))?;

        info!("Safetensors file saved: {}", output_path.display());
        Ok(())
    }
}

/// Convenience function to run convert command
pub fn run_convert_command(args: ConvertArgs) -> Result<()> {
    let manifest_dir = match &args.command {
        ConvertCommand::FromHub(args) => args.manifest_dir.clone(),
        ConvertCommand::List(args) => args.manifest_dir.clone(),
        ConvertCommand::Delete(args) => args.manifest_dir.clone(),
        ConvertCommand::FromLocal(_) => None,
    };

    let handler = ConvertHandler::new(manifest_dir)?;
    handler.handle_command(args.command)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_convert_handler_creation() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let handler = ConvertHandler::new(Some(temp_dir.path().to_path_buf()))?;

        // Basic functionality test
        assert!(handler.manifest_manager.list_manifests().is_ok());

        Ok(())
    }

    #[test]
    fn test_save_tensors_as_safetensors() -> Result<()> {
        use candle_core::{Device, DType, Tensor};
        use std::collections::HashMap;

        let temp_dir = TempDir::new()?;
        let handler = ConvertHandler::new(Some(temp_dir.path().to_path_buf()))?;

        // Create test tensors
        let mut tensors = HashMap::new();
        let test_tensor = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;
        tensors.insert("test_weight".to_string(), test_tensor);

        let output_path = temp_dir.path().join("test.safetensors");
        handler.save_tensors_as_safetensors(&tensors, &output_path)?;

        assert!(output_path.exists());
        assert!(output_path.metadata()?.len() > 0);

        Ok(())
    }
}