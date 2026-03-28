use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};
use turbocalm_core::{AutoencoderConfig, CALMConfig};

/// Manifest for a converted CALM model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CALMModelManifest {
    /// Manifest format version
    pub version: String,

    /// Model metadata
    pub metadata: ModelMetadata,

    /// CALM configuration
    pub calm_config: CALMConfig,

    /// Autoencoder configuration (optional)
    pub autoencoder_config: Option<AutoencoderConfig>,

    /// File information
    pub files: FileManifest,

    /// Conversion information
    pub conversion: ConversionInfo,

    /// Verification results
    pub verification: Option<VerificationSummary>,
}

impl CALMModelManifest {
    /// Current manifest version
    pub const CURRENT_VERSION: &'static str = "1.0.0";

    /// Create a new manifest
    pub fn new(
        model_id: &str,
        calm_config: CALMConfig,
        autoencoder_config: Option<AutoencoderConfig>,
    ) -> Self {
        Self {
            version: Self::CURRENT_VERSION.to_string(),
            metadata: ModelMetadata {
                model_id: model_id.to_string(),
                model_name: extract_model_name(model_id),
                created_at: chrono::Utc::now().to_rfc3339(),
                description: None,
                tags: Vec::new(),
                original_source: Some(model_id.to_string()),
            },
            calm_config,
            autoencoder_config,
            files: FileManifest::default(),
            conversion: ConversionInfo::default(),
            verification: None,
        }
    }

    /// Load manifest from file
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(&path)?;
        let manifest: Self = serde_json::from_str(&content)?;

        // Validate version compatibility
        if manifest.version != Self::CURRENT_VERSION {
            warn!(
                "Manifest version {} may be incompatible with current version {}",
                manifest.version,
                Self::CURRENT_VERSION
            );
        }

        debug!("Loaded manifest from: {}", path.as_ref().display());
        Ok(manifest)
    }

    /// Save manifest to file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(&path, content)?;
        info!("Saved manifest to: {}", path.as_ref().display());
        Ok(())
    }

    /// Add a safetensors file to the manifest
    pub fn add_safetensors_file(&mut self, filename: &str, path: &Path, checksum: Option<String>) {
        let file_info = FileInfo {
            path: path.to_path_buf(),
            size_bytes: std::fs::metadata(path)
                .map(|m| m.len())
                .unwrap_or(0),
            checksum,
            created_at: chrono::Utc::now().to_rfc3339(),
            file_type: FileType::SafeTensors,
        };

        self.files.model_files.insert(filename.to_string(), file_info);
    }

    /// Add a config file to the manifest
    pub fn add_config_file(&mut self, filename: &str, path: &Path) {
        let file_info = FileInfo {
            path: path.to_path_buf(),
            size_bytes: std::fs::metadata(path)
                .map(|m| m.len())
                .unwrap_or(0),
            checksum: None,
            created_at: chrono::Utc::now().to_rfc3339(),
            file_type: FileType::Config,
        };

        self.files.config_files.insert(filename.to_string(), file_info);
    }

    /// Add tokenizer files to the manifest
    pub fn add_tokenizer_file(&mut self, filename: &str, path: &Path) {
        let file_info = FileInfo {
            path: path.to_path_buf(),
            size_bytes: std::fs::metadata(path)
                .map(|m| m.len())
                .unwrap_or(0),
            checksum: None,
            created_at: chrono::Utc::now().to_rfc3339(),
            file_type: FileType::Tokenizer,
        };

        self.files.tokenizer_files.insert(filename.to_string(), file_info);
    }

    /// Set conversion information
    pub fn set_conversion_info(&mut self, info: ConversionInfo) {
        self.conversion = info;
    }

    /// Set verification results
    pub fn set_verification_results(&mut self, verification: VerificationSummary) {
        self.verification = Some(verification);
    }

    /// Get total model size in bytes
    pub fn total_size_bytes(&self) -> u64 {
        self.files.model_files.values()
            .map(|f| f.size_bytes)
            .sum::<u64>()
    }

    /// Get total model size in MB
    pub fn total_size_mb(&self) -> f64 {
        self.total_size_bytes() as f64 / (1024.0 * 1024.0)
    }

    /// Get model parameter count estimate
    pub fn estimated_parameters(&self) -> Option<u64> {
        self.conversion.parameter_count
    }

    /// Check if all files exist
    pub fn validate_file_paths(&self) -> Vec<String> {
        let mut missing_files = Vec::new();

        // Check model files
        for (name, info) in &self.files.model_files {
            if !info.path.exists() {
                missing_files.push(format!("Model file '{}': {}", name, info.path.display()));
            }
        }

        // Check config files
        for (name, info) in &self.files.config_files {
            if !info.path.exists() {
                missing_files.push(format!("Config file '{}': {}", name, info.path.display()));
            }
        }

        // Check tokenizer files
        for (name, info) in &self.files.tokenizer_files {
            if !info.path.exists() {
                missing_files.push(format!("Tokenizer file '{}': {}", name, info.path.display()));
            }
        }

        missing_files
    }

    /// Display manifest summary
    pub fn display_summary(&self) {
        info!("CALM Model Manifest Summary:");
        info!("  Model: {}", self.metadata.model_id);
        info!("  Version: {}", self.version);
        info!("  Created: {}", self.metadata.created_at);
        info!("  Total size: {:.1} MB", self.total_size_mb());

        if let Some(param_count) = self.estimated_parameters() {
            info!("  Parameters: {:.2}M", param_count as f64 / 1_000_000.0);
        }

        info!("  Files:");
        info!("    Model files: {}", self.files.model_files.len());
        info!("    Config files: {}", self.files.config_files.len());
        info!("    Tokenizer files: {}", self.files.tokenizer_files.len());

        if let Some(ref verification) = self.verification {
            info!("  Verification: {} (errors: {}, warnings: {})",
                if verification.passed { "PASSED" } else { "FAILED" },
                verification.error_count,
                verification.warning_count
            );
        }
    }
}

/// Model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub model_id: String,
    pub model_name: String,
    pub created_at: String,
    pub description: Option<String>,
    pub tags: Vec<String>,
    pub original_source: Option<String>,
}

/// File manifest containing all model files
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FileManifest {
    pub model_files: HashMap<String, FileInfo>,
    pub config_files: HashMap<String, FileInfo>,
    pub tokenizer_files: HashMap<String, FileInfo>,
}

/// Information about a single file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileInfo {
    pub path: PathBuf,
    pub size_bytes: u64,
    pub checksum: Option<String>,
    pub created_at: String,
    pub file_type: FileType,
}

/// Type of file in the manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FileType {
    SafeTensors,
    Config,
    Tokenizer,
    Other(String),
}

/// Conversion information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversionInfo {
    pub converter_version: String,
    pub conversion_date: String,
    pub source_format: String,
    pub target_format: String,
    pub parameter_count: Option<u64>,
    pub conversion_options: HashMap<String, serde_json::Value>,
}

impl Default for ConversionInfo {
    fn default() -> Self {
        Self {
            converter_version: env!("CARGO_PKG_VERSION").to_string(),
            conversion_date: chrono::Utc::now().to_rfc3339(),
            source_format: "unknown".to_string(),
            target_format: "safetensors".to_string(),
            parameter_count: None,
            conversion_options: HashMap::new(),
        }
    }
}

/// Summary of verification results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationSummary {
    pub passed: bool,
    pub total_tensors: usize,
    pub error_count: usize,
    pub warning_count: usize,
    pub verification_date: String,
}

impl VerificationSummary {
    /// Create from verification report
    pub fn from_verification_report(report: &crate::verification::VerificationReport) -> Self {
        Self {
            passed: report.passed(),
            total_tensors: report.total_tensors,
            error_count: report.error_count,
            warning_count: report.warning_count,
            verification_date: chrono::Utc::now().to_rfc3339(),
        }
    }
}

/// Extract model name from model ID
fn extract_model_name(model_id: &str) -> String {
    if let Some(name) = model_id.split('/').last() {
        name.to_string()
    } else {
        model_id.to_string()
    }
}

/// Manifest manager for handling multiple models
pub struct ManifestManager {
    manifest_dir: PathBuf,
}

impl ManifestManager {
    /// Create a new manifest manager
    pub fn new<P: AsRef<Path>>(manifest_dir: P) -> Result<Self> {
        let manifest_dir = manifest_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&manifest_dir)?;

        Ok(Self { manifest_dir })
    }

    /// Save a manifest for a model
    pub fn save_manifest(&self, manifest: &CALMModelManifest) -> Result<PathBuf> {
        let filename = format!("{}.manifest.json",
            sanitize_filename(&manifest.metadata.model_name));
        let path = self.manifest_dir.join(filename);

        manifest.save_to_file(&path)?;
        Ok(path)
    }

    /// Load a manifest for a model
    pub fn load_manifest(&self, model_name: &str) -> Result<CALMModelManifest> {
        let filename = format!("{}.manifest.json", sanitize_filename(model_name));
        let path = self.manifest_dir.join(filename);

        CALMModelManifest::load_from_file(path)
    }

    /// List all available manifests
    pub fn list_manifests(&self) -> Result<Vec<ModelSummary>> {
        let mut summaries = Vec::new();

        for entry in std::fs::read_dir(&self.manifest_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("json")
                && path.file_stem()
                    .and_then(|s| s.to_str())
                    .map(|s| s.ends_with(".manifest"))
                    .unwrap_or(false)
            {
                match CALMModelManifest::load_from_file(&path) {
                    Ok(manifest) => {
                        summaries.push(ModelSummary {
                            model_id: manifest.metadata.model_id,
                            model_name: manifest.metadata.model_name,
                            size_mb: manifest.total_size_mb(),
                            created_at: manifest.metadata.created_at,
                            manifest_path: path,
                        });
                    }
                    Err(e) => {
                        warn!("Failed to load manifest from {}: {}", path.display(), e);
                    }
                }
            }
        }

        summaries.sort_by(|a, b| b.created_at.cmp(&a.created_at)); // Most recent first
        Ok(summaries)
    }

    /// Delete a manifest and optionally the model files
    pub fn delete_manifest(&self, model_name: &str, delete_files: bool) -> Result<()> {
        let manifest = self.load_manifest(model_name)?;

        if delete_files {
            info!("Deleting model files for: {}", model_name);

            // Delete model files
            for (_, file_info) in &manifest.files.model_files {
                if file_info.path.exists() {
                    std::fs::remove_file(&file_info.path)?;
                    debug!("Deleted: {}", file_info.path.display());
                }
            }

            // Delete config files
            for (_, file_info) in &manifest.files.config_files {
                if file_info.path.exists() {
                    std::fs::remove_file(&file_info.path)?;
                    debug!("Deleted: {}", file_info.path.display());
                }
            }

            // Delete tokenizer files
            for (_, file_info) in &manifest.files.tokenizer_files {
                if file_info.path.exists() {
                    std::fs::remove_file(&file_info.path)?;
                    debug!("Deleted: {}", file_info.path.display());
                }
            }
        }

        // Delete manifest file
        let filename = format!("{}.manifest.json", sanitize_filename(model_name));
        let manifest_path = self.manifest_dir.join(filename);
        if manifest_path.exists() {
            std::fs::remove_file(manifest_path)?;
            info!("Deleted manifest for: {}", model_name);
        }

        Ok(())
    }
}

/// Summary information about a model
#[derive(Debug, Clone)]
pub struct ModelSummary {
    pub model_id: String,
    pub model_name: String,
    pub size_mb: f64,
    pub created_at: String,
    pub manifest_path: PathBuf,
}

impl ModelSummary {
    /// Display summary information
    pub fn display(&self) {
        info!("  {}: {:.1} MB ({})",
            self.model_name,
            self.size_mb,
            self.created_at
        );
    }
}

/// Sanitize filename for cross-platform compatibility
fn sanitize_filename(name: &str) -> String {
    name.chars()
        .map(|c| match c {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
            c => c,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_manifest_creation() {
        let calm_config = CALMConfig::default();
        let mut manifest = CALMModelManifest::new(
            "test/model",
            calm_config,
            None,
        );

        assert_eq!(manifest.metadata.model_id, "test/model");
        assert_eq!(manifest.metadata.model_name, "model");
        assert_eq!(manifest.version, CALMModelManifest::CURRENT_VERSION);

        // Test adding files
        let temp_path = PathBuf::from("/tmp/test.safetensors");
        manifest.add_safetensors_file("model.safetensors", &temp_path, None);
        assert_eq!(manifest.files.model_files.len(), 1);
    }

    #[test]
    fn test_manifest_serialization() {
        let calm_config = CALMConfig::default();
        let manifest = CALMModelManifest::new(
            "test/model",
            calm_config,
            None,
        );

        // Test JSON serialization
        let json = serde_json::to_string(&manifest).unwrap();
        let parsed: CALMModelManifest = serde_json::from_str(&json).unwrap();

        assert_eq!(manifest.metadata.model_id, parsed.metadata.model_id);
        assert_eq!(manifest.version, parsed.version);
    }

    #[test]
    fn test_manifest_manager() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let manager = ManifestManager::new(temp_dir.path())?;

        let calm_config = CALMConfig::default();
        let manifest = CALMModelManifest::new(
            "test/model",
            calm_config,
            None,
        );

        // Save manifest
        let saved_path = manager.save_manifest(&manifest)?;
        assert!(saved_path.exists());

        // Load manifest
        let loaded_manifest = manager.load_manifest("model")?;
        assert_eq!(loaded_manifest.metadata.model_id, manifest.metadata.model_id);

        // List manifests
        let summaries = manager.list_manifests()?;
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].model_name, "model");

        Ok(())
    }

    #[test]
    fn test_filename_sanitization() {
        assert_eq!(sanitize_filename("test/model"), "test_model");
        assert_eq!(sanitize_filename("model:name"), "model_name");
        assert_eq!(sanitize_filename("normal_name"), "normal_name");
    }

    #[test]
    fn test_model_name_extraction() {
        assert_eq!(extract_model_name("org/model-name"), "model-name");
        assert_eq!(extract_model_name("simple"), "simple");
        assert_eq!(extract_model_name("a/b/c"), "c");
    }

    #[test]
    fn test_verification_summary() {
        // This would normally come from an actual verification report
        let summary = VerificationSummary {
            passed: true,
            total_tensors: 100,
            error_count: 0,
            warning_count: 2,
            verification_date: chrono::Utc::now().to_rfc3339(),
        };

        assert!(summary.passed);
        assert_eq!(summary.total_tensors, 100);
        assert_eq!(summary.error_count, 0);
        assert_eq!(summary.warning_count, 2);
    }
}