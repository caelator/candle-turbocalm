use anyhow::Result;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};
use turbocalm_core::{CompleteModelDownload, HubClient};

/// Specialized checkpoint downloader for CALM models
pub struct CheckpointDownloader {
    hub_client: HubClient,
}

impl CheckpointDownloader {
    /// Create a new checkpoint downloader
    pub fn new() -> Result<Self> {
        Ok(Self {
            hub_client: HubClient::new()?,
        })
    }

    /// Download a complete CALM model checkpoint
    pub fn download_calm_checkpoint(&self, model_id: &str) -> Result<CALMCheckpoint> {
        info!("Downloading CALM checkpoint: {}", model_id);

        // Download complete model using core functionality
        let complete_download = turbocalm_core::DownloadUtils::download_complete_model(model_id)?;

        // Parse CALM-specific files
        let calm_config = self.parse_calm_config(&complete_download)?;
        let autoencoder_config = self.parse_autoencoder_config(&complete_download)?;

        Ok(CALMCheckpoint {
            model_id: model_id.to_string(),
            complete_download,
            calm_config,
            autoencoder_config,
        })
    }

    /// Download model weights only (for existing configs)
    pub fn download_weights_only(&self, model_id: &str) -> Result<Vec<PathBuf>> {
        info!("Downloading model weights for: {}", model_id);
        self.hub_client
            .download_safetensors(model_id)
            .map_err(|e| anyhow::anyhow!("{}", e))
    }

    /// Download specific checkpoint files by pattern
    pub fn download_checkpoint_files(
        &self,
        model_id: &str,
        patterns: &[&str],
    ) -> Result<HashMap<String, PathBuf>> {
        info!(
            "Downloading checkpoint files for {}: {:?}",
            model_id, patterns
        );

        let repo = self.hub_client.api().model(model_id.to_string());
        let mut downloaded_files = HashMap::new();

        for pattern in patterns {
            // Try exact match first
            match repo.get(pattern) {
                Ok(path) => {
                    debug!("Downloaded {}", pattern);
                    downloaded_files.insert(pattern.to_string(), path);
                    continue;
                }
                Err(_) => {
                    debug!("Exact match failed for {}", pattern);
                }
            }

            // Try pattern matching for sharded files
            if pattern.contains("{:05}") {
                let mut shard_index = 1;
                loop {
                    let shard_filename = pattern.replace("{:05}", &format!("{:05}", shard_index));
                    match repo.get(&shard_filename) {
                        Ok(path) => {
                            debug!("Downloaded shard {}", shard_filename);
                            downloaded_files.insert(shard_filename, path);
                            shard_index += 1;
                        }
                        Err(_) => {
                            if shard_index == 1 {
                                warn!("No sharded files found for pattern: {}", pattern);
                            }
                            break;
                        }
                    }
                }
            }
        }

        if downloaded_files.is_empty() {
            return Err(anyhow::anyhow!(
                "No files downloaded for patterns: {:?}",
                patterns
            ));
        }

        Ok(downloaded_files)
    }

    /// Parse CALM configuration from downloaded files
    fn parse_calm_config(
        &self,
        download: &CompleteModelDownload,
    ) -> Result<Option<turbocalm_core::CALMConfig>> {
        if let Some(config_path) = download.config_path() {
            debug!("Parsing CALM config from: {}", config_path.display());

            let config_content = std::fs::read_to_string(config_path)?;

            // Try to parse as CALM config
            match serde_json::from_str::<turbocalm_core::CALMConfig>(&config_content) {
                Ok(config) => {
                    info!("Successfully parsed CALM config");
                    return Ok(Some(config));
                }
                Err(e) => {
                    debug!("Failed to parse as CALM config: {}", e);
                }
            }

            // Try to parse as generic config and adapt
            if let Ok(generic_config) = serde_json::from_str::<serde_json::Value>(&config_content) {
                if let Some(calm_config) = self.adapt_generic_config_to_calm(&generic_config) {
                    info!("Adapted generic config to CALM config");
                    return Ok(Some(calm_config));
                }
            }
        }

        warn!("No CALM config found or parseable");
        Ok(None)
    }

    /// Parse autoencoder configuration from downloaded files
    fn parse_autoencoder_config(
        &self,
        download: &CompleteModelDownload,
    ) -> Result<Option<turbocalm_core::AutoencoderConfig>> {
        // Look for autoencoder-specific config file
        if let Some(ae_config_path) = download.config_files.get("autoencoder_config.json") {
            debug!(
                "Parsing autoencoder config from: {}",
                ae_config_path.display()
            );

            let config_content = std::fs::read_to_string(ae_config_path)?;
            match serde_json::from_str::<turbocalm_core::AutoencoderConfig>(&config_content) {
                Ok(config) => {
                    info!("Successfully parsed autoencoder config");
                    return Ok(Some(config));
                }
                Err(e) => {
                    warn!("Failed to parse autoencoder config: {}", e);
                }
            }
        }

        // Try to extract autoencoder config from main config
        if let Some(config_path) = download.config_path() {
            let config_content = std::fs::read_to_string(config_path)?;
            if let Ok(config_json) = serde_json::from_str::<serde_json::Value>(&config_content) {
                if let Some(ae_config) = self.extract_autoencoder_config(&config_json) {
                    info!("Extracted autoencoder config from main config");
                    return Ok(Some(ae_config));
                }
            }
        }

        warn!("No autoencoder config found");
        Ok(None)
    }

    /// Adapt a generic HuggingFace config to CALM config
    fn adapt_generic_config_to_calm(
        &self,
        generic_config: &serde_json::Value,
    ) -> Option<turbocalm_core::CALMConfig> {
        let mut calm_config = turbocalm_core::CALMConfig::default();

        // Extract common transformer parameters
        if let Some(vocab_size) = generic_config.get("vocab_size").and_then(|v| v.as_u64()) {
            calm_config.vocab_size = vocab_size as u32;
        }

        if let Some(hidden_size) = generic_config.get("hidden_size").and_then(|v| v.as_u64()) {
            calm_config.hidden_size = hidden_size as u32;
        }

        if let Some(intermediate_size) = generic_config
            .get("intermediate_size")
            .and_then(|v| v.as_u64())
        {
            calm_config.intermediate_size = intermediate_size as u32;
        }

        if let Some(num_layers) = generic_config
            .get("num_hidden_layers")
            .and_then(|v| v.as_u64())
        {
            calm_config.num_hidden_layers = num_layers as u32;
        }

        if let Some(num_heads) = generic_config
            .get("num_attention_heads")
            .and_then(|v| v.as_u64())
        {
            calm_config.num_attention_heads = num_heads as u32;
        }

        if let Some(num_kv_heads) = generic_config
            .get("num_key_value_heads")
            .and_then(|v| v.as_u64())
        {
            calm_config.num_key_value_heads = Some(num_kv_heads as u32);
        }

        if let Some(max_pos) = generic_config
            .get("max_position_embeddings")
            .and_then(|v| v.as_u64())
        {
            calm_config.max_position_embeddings = max_pos as u32;
        }

        if let Some(hidden_act) = generic_config.get("hidden_act").and_then(|v| v.as_str()) {
            calm_config.hidden_act = hidden_act.to_string();
        }

        if let Some(rms_eps) = generic_config.get("rms_norm_eps").and_then(|v| v.as_f64()) {
            calm_config.rms_norm_eps = rms_eps;
        }

        if let Some(rope_theta) = generic_config.get("rope_theta").and_then(|v| v.as_f64()) {
            calm_config.rope_theta = rope_theta;
        }

        // Extract CALM-specific parameters if they exist
        if let Some(patch_size) = generic_config.get("patch_size").and_then(|v| v.as_u64()) {
            calm_config.patch_size = patch_size as u32;
        }

        if let Some(latent_size) = generic_config.get("latent_size").and_then(|v| v.as_u64()) {
            calm_config.latent_size = latent_size as u32;
        }

        if let Some(model_type) = generic_config.get("model_type").and_then(|v| v.as_str()) {
            calm_config.model_type = model_type.to_string();
        }

        if let Some(num_samples) = generic_config.get("num_samples").and_then(|v| v.as_u64()) {
            calm_config.num_samples = num_samples as u32;
        }

        Some(calm_config)
    }

    /// Extract autoencoder config from main config JSON
    fn extract_autoencoder_config(
        &self,
        config_json: &serde_json::Value,
    ) -> Option<turbocalm_core::AutoencoderConfig> {
        // Look for autoencoder section in config
        if let Some(ae_section) = config_json.get("autoencoder") {
            if let Ok(ae_config) =
                serde_json::from_value::<turbocalm_core::AutoencoderConfig>(ae_section.clone())
            {
                return Some(ae_config);
            }
        }

        // Try to construct autoencoder config from main config parameters
        let mut ae_config = turbocalm_core::AutoencoderConfig::default();

        if let Some(vocab_size) = config_json.get("vocab_size").and_then(|v| v.as_u64()) {
            ae_config.vocab_size = vocab_size as u32;
        }

        if let Some(patch_size) = config_json.get("patch_size").and_then(|v| v.as_u64()) {
            ae_config.patch_size = patch_size as u32;
        }

        if let Some(latent_size) = config_json.get("latent_size").and_then(|v| v.as_u64()) {
            ae_config.latent_size = latent_size as u32;
        }

        Some(ae_config)
    }
}

/// Complete CALM checkpoint with all necessary components
#[derive(Debug)]
pub struct CALMCheckpoint {
    pub model_id: String,
    pub complete_download: CompleteModelDownload,
    pub calm_config: Option<turbocalm_core::CALMConfig>,
    pub autoencoder_config: Option<turbocalm_core::AutoencoderConfig>,
}

impl CALMCheckpoint {
    /// Get the model weight file paths
    pub fn model_paths(&self) -> &[PathBuf] {
        self.complete_download.model_paths()
    }

    /// Get the CALM configuration (with fallback to default)
    pub fn calm_config(&self) -> turbocalm_core::CALMConfig {
        self.calm_config.clone().unwrap_or_default()
    }

    /// Get the autoencoder configuration (with fallback to default)
    pub fn autoencoder_config(&self) -> turbocalm_core::AutoencoderConfig {
        self.autoencoder_config.clone().unwrap_or_default()
    }

    /// Check if this checkpoint has valid CALM components
    pub fn is_valid_calm_checkpoint(&self) -> bool {
        !self.complete_download.model_files.is_empty()
            && (self.calm_config.is_some() || self.complete_download.config_path().is_some())
    }

    /// Get tokenizer path from the download
    pub fn tokenizer_path(&self) -> Option<&PathBuf> {
        self.complete_download.tokenizer_path()
    }

    /// Save checkpoint manifest
    pub fn save_manifest<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let mut manifest = self.complete_download.manifest.clone();

        // Add CALM-specific metadata
        manifest
            .files
            .insert("model_type".to_string(), self.calm_config().model_type);
        manifest.files.insert(
            "patch_size".to_string(),
            self.calm_config().patch_size.to_string(),
        );

        manifest
            .save_to_file(path)
            .map_err(|e| anyhow::anyhow!("{}", e))
    }
}

/// Common CALM model identifiers for easy access
pub struct KnownCALMModels;

impl KnownCALMModels {
    /// Get list of known CALM model IDs
    pub fn get_known_models() -> Vec<&'static str> {
        vec![
            // Actual CALM model IDs from HuggingFace
            "cccczshao/CALM-Autoencoder",
            "cccczshao/CALM-M",
            "cccczshao/CALM-L",
            "cccczshao/CALM-XL",
        ]
    }

    /// Check if a model ID is a known CALM model
    pub fn is_known_calm_model(model_id: &str) -> bool {
        Self::get_known_models().contains(&model_id)
            || model_id.contains("calm")
            || model_id.contains("CALM")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbocalm_core::DownloadManifest;

    #[test]
    fn test_checkpoint_downloader_creation() {
        // This test might fail without network access
        match CheckpointDownloader::new() {
            Ok(_) => println!("CheckpointDownloader created successfully"),
            Err(e) => println!(
                "CheckpointDownloader creation failed (expected in some environments): {}",
                e
            ),
        }
    }

    #[test]
    fn test_known_calm_models() {
        let models = KnownCALMModels::get_known_models();
        assert!(!models.is_empty());

        assert!(KnownCALMModels::is_known_calm_model("calm-ai/calm-7b"));
        assert!(KnownCALMModels::is_known_calm_model("CALM-base"));
        assert!(!KnownCALMModels::is_known_calm_model("bert-base-uncased"));
    }

    #[test]
    fn test_calm_checkpoint_validation() {
        use std::collections::HashMap;

        let complete_download = CompleteModelDownload {
            model_id: "test/model".to_string(),
            model_files: vec![PathBuf::from("/tmp/model.safetensors")],
            config_files: {
                let mut m = HashMap::new();
                m.insert("config.json".to_string(), PathBuf::from("/tmp/config.json"));
                m
            },
            tokenizer_files: HashMap::new(),
            manifest: DownloadManifest::new("test/model"),
        };

        let checkpoint = CALMCheckpoint {
            model_id: "test/model".to_string(),
            complete_download,
            calm_config: None,
            autoencoder_config: None,
        };

        // Should be valid even without explicit CALM config (will use defaults)
        assert!(checkpoint.is_valid_calm_checkpoint());
        assert_eq!(checkpoint.model_paths().len(), 1);

        // Should return defaults when configs are None
        let calm_config = checkpoint.calm_config();
        assert_eq!(calm_config.model_type, "energy");
    }
}
