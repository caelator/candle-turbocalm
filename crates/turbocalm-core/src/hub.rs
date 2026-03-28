use crate::error::{HubError, Result, TurboCALMError};
use hf_hub::api::sync::Api;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

/// HuggingFace Hub client for downloading model files
pub struct HubClient {
    api: Api,
}

impl HubClient {
    /// Create a new HuggingFace Hub client
    pub fn new() -> Result<Self> {
        let api = Api::new().map_err(|e| {
            TurboCALMError::Hub(HubError::NetworkError(format!(
                "Failed to initialize HF API: {}",
                e
            )))
        })?;

        Ok(Self { api })
    }

    /// Download a specific file from a model repository
    pub fn download_file(&self, model_id: &str, filename: &str) -> Result<PathBuf> {
        info!("Downloading {} from {}", filename, model_id);

        let repo = self.api.model(model_id.to_string());
        repo.get(filename).map_err(|e| {
            TurboCALMError::Hub(HubError::DownloadFailed(format!(
                "Failed to download {} from {}: {}",
                filename, model_id, e
            )))
        })
    }

    /// Download multiple files from a model repository
    pub fn download_files(&self, model_id: &str, filenames: &[&str]) -> Result<Vec<PathBuf>> {
        info!("Downloading {} files from {}", filenames.len(), model_id);

        let repo = self.api.model(model_id.to_string());
        let mut paths = Vec::new();

        for filename in filenames {
            debug!("Downloading {}", filename);
            let path = repo.get(filename).map_err(|e| {
                TurboCALMError::Hub(HubError::DownloadFailed(format!(
                    "Failed to download {} from {}: {}",
                    filename, model_id, e
                )))
            })?;
            paths.push(path);
        }

        Ok(paths)
    }

    /// Download all safetensors files from a model
    pub fn download_safetensors(&self, model_id: &str) -> Result<Vec<PathBuf>> {
        info!("Downloading safetensors files from {}", model_id);

        // Common safetensors file patterns
        let common_patterns = [
            "model.safetensors",
            "pytorch_model.safetensors",
        ];

        let repo = self.api.model(model_id.to_string());
        let mut downloaded_files = Vec::new();

        // Try to download common single-file patterns first
        for pattern in &common_patterns {
            match repo.get(pattern) {
                Ok(path) => {
                    debug!("Downloaded {}", pattern);
                    downloaded_files.push(path);
                    return Ok(downloaded_files);
                }
                Err(_) => {
                    debug!("{} not found, trying next pattern", pattern);
                }
            }
        }

        // Try sharded safetensors files
        for i in 1..=20 {
            // Try up to 20 shards
            let filename = format!("model-{:05}-of-{:05}.safetensors", i, i);
            match repo.get(&filename) {
                Ok(path) => {
                    debug!("Downloaded shard {}", filename);
                    downloaded_files.push(path);
                }
                Err(_) => {
                    if i == 1 {
                        // No sharded files found
                        break;
                    }
                }
            }
        }

        if downloaded_files.is_empty() {
            return Err(TurboCALMError::Hub(HubError::DownloadFailed(
                format!("No safetensors files found in {}", model_id),
            )));
        }

        Ok(downloaded_files)
    }

    /// Download configuration files (config.json, generation_config.json, etc.)
    pub fn download_config_files(&self, model_id: &str) -> Result<HashMap<String, PathBuf>> {
        info!("Downloading configuration files from {}", model_id);

        let config_files = [
            "config.json",
            "generation_config.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ];

        let repo = self.api.model(model_id.to_string());
        let mut configs = HashMap::new();

        for config_file in &config_files {
            match repo.get(config_file) {
                Ok(path) => {
                    debug!("Downloaded {}", config_file);
                    configs.insert(config_file.to_string(), path);
                }
                Err(_) => {
                    debug!("{} not available", config_file);
                }
            }
        }

        if configs.is_empty() {
            warn!("No configuration files found in {}", model_id);
        }

        Ok(configs)
    }

    /// Download tokenizer files
    pub fn download_tokenizer_files(&self, model_id: &str) -> Result<HashMap<String, PathBuf>> {
        info!("Downloading tokenizer files from {}", model_id);

        let tokenizer_files = [
            "tokenizer.json",
            "tokenizer.model",
            "vocab.txt",
            "merges.txt",
            "added_tokens.json",
        ];

        let repo = self.api.model(model_id.to_string());
        let mut files = HashMap::new();

        for file in &tokenizer_files {
            match repo.get(file) {
                Ok(path) => {
                    debug!("Downloaded {}", file);
                    files.insert(file.to_string(), path);
                }
                Err(_) => {
                    debug!("{} not available", file);
                }
            }
        }

        if files.is_empty() {
            return Err(TurboCALMError::Hub(HubError::DownloadFailed(
                format!("No tokenizer files found in {}", model_id),
            )));
        }

        Ok(files)
    }

    /// Check if a model exists on the hub (by trying to get README.md)
    pub fn model_exists(&self, model_id: &str) -> bool {
        info!("Checking if model {} exists", model_id);

        let repo = self.api.model(model_id.to_string());
        match repo.get("README.md") {
            Ok(_) => {
                debug!("Model {} exists", model_id);
                true
            }
            Err(_) => {
                // Try config.json as fallback
                match repo.get("config.json") {
                    Ok(_) => {
                        debug!("Model {} exists (found config.json)", model_id);
                        true
                    }
                    Err(_) => {
                        debug!("Model {} does not exist or is private", model_id);
                        false
                    }
                }
            }
        }
    }

    /// Get the HuggingFace API instance
    pub fn api(&self) -> &Api {
        &self.api
    }
}

impl Default for HubClient {
    fn default() -> Self {
        Self::new().expect("Failed to create default HubClient")
    }
}

/// Download manifest for tracking downloaded files
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DownloadManifest {
    pub model_id: String,
    pub downloaded_at: String,
    pub files: HashMap<String, String>, // filename -> local_path
}

impl DownloadManifest {
    /// Create a new download manifest
    pub fn new(model_id: &str) -> Self {
        Self {
            model_id: model_id.to_string(),
            downloaded_at: chrono::Utc::now().to_rfc3339(),
            files: HashMap::new(),
        }
    }

    /// Add a downloaded file to the manifest
    pub fn add_file(&mut self, filename: &str, local_path: &Path) {
        self.files
            .insert(filename.to_string(), local_path.to_string_lossy().to_string());
    }

    /// Save manifest to a JSON file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Load manifest from a JSON file
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let manifest: Self = serde_json::from_str(&content)?;
        Ok(manifest)
    }
}

/// High-level download utilities
pub struct DownloadUtils;

impl DownloadUtils {
    /// Download a complete model (safetensors + config + tokenizer)
    pub fn download_complete_model(model_id: &str) -> Result<CompleteModelDownload> {
        let client = HubClient::new()?;

        info!("Downloading complete model: {}", model_id);

        // Check if model exists
        if !client.model_exists(model_id) {
            return Err(TurboCALMError::Hub(HubError::ModelNotFound(
                model_id.to_string(),
            )));
        }

        // Download model weights (safetensors)
        let model_files = client.download_safetensors(model_id)?;

        // Download configuration files
        let config_files = client.download_config_files(model_id)?;

        // Download tokenizer files
        let tokenizer_files = client.download_tokenizer_files(model_id)?;

        // Create manifest
        let mut manifest = DownloadManifest::new(model_id);
        for (i, path) in model_files.iter().enumerate() {
            manifest.add_file(&format!("model_{}.safetensors", i), path);
        }
        for (name, path) in &config_files {
            manifest.add_file(name, path);
        }
        for (name, path) in &tokenizer_files {
            manifest.add_file(name, path);
        }

        Ok(CompleteModelDownload {
            model_id: model_id.to_string(),
            model_files,
            config_files,
            tokenizer_files,
            manifest,
        })
    }
}

/// Result of downloading a complete model
#[derive(Debug)]
pub struct CompleteModelDownload {
    pub model_id: String,
    pub model_files: Vec<PathBuf>,
    pub config_files: HashMap<String, PathBuf>,
    pub tokenizer_files: HashMap<String, PathBuf>,
    pub manifest: DownloadManifest,
}

impl CompleteModelDownload {
    /// Get the main config file path
    pub fn config_path(&self) -> Option<&PathBuf> {
        self.config_files.get("config.json")
    }

    /// Get the tokenizer path
    pub fn tokenizer_path(&self) -> Option<&PathBuf> {
        self.tokenizer_files
            .get("tokenizer.json")
            .or_else(|| self.tokenizer_files.get("tokenizer.model"))
    }

    /// Get all model weight file paths
    pub fn model_paths(&self) -> &[PathBuf] {
        &self.model_files
    }
}

/// Convenience functions
pub mod convenience {
    use super::*;

    /// Quick function to download a single file
    pub fn download_file(model_id: &str, filename: &str) -> Result<PathBuf> {
        HubClient::new()?.download_file(model_id, filename)
    }

    /// Quick function to download model config
    pub fn download_config(model_id: &str) -> Result<PathBuf> {
        download_file(model_id, "config.json")
    }

    /// Quick function to download tokenizer
    pub fn download_tokenizer(model_id: &str) -> Result<PathBuf> {
        download_file(model_id, "tokenizer.json")
    }

    /// Quick function to check if model exists
    pub fn model_exists(model_id: &str) -> bool {
        HubClient::new()
            .map(|client| client.model_exists(model_id))
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_download_manifest() {
        let mut manifest = DownloadManifest::new("test/model");
        assert_eq!(manifest.model_id, "test/model");
        assert!(manifest.files.is_empty());

        manifest.add_file("config.json", Path::new("/tmp/config.json"));
        assert_eq!(manifest.files.len(), 1);
        assert!(manifest.files.contains_key("config.json"));
    }

    #[test]
    fn test_hub_client_creation() {
        // This test might fail if there's no network connection or HF API issues
        let result = HubClient::new();
        match result {
            Ok(_) => println!("HubClient created successfully"),
            Err(e) => println!("HubClient creation failed (expected in some environments): {}", e),
        }
    }

    #[test]
    fn test_complete_model_download_accessors() {
        use std::collections::HashMap;

        let mut config_files = HashMap::new();
        config_files.insert("config.json".to_string(), PathBuf::from("/tmp/config.json"));

        let mut tokenizer_files = HashMap::new();
        tokenizer_files.insert("tokenizer.json".to_string(), PathBuf::from("/tmp/tokenizer.json"));

        let download = CompleteModelDownload {
            model_id: "test/model".to_string(),
            model_files: vec![PathBuf::from("/tmp/model.safetensors")],
            config_files,
            tokenizer_files,
            manifest: DownloadManifest::new("test/model"),
        };

        assert_eq!(download.model_id, "test/model");
        assert!(download.config_path().is_some());
        assert!(download.tokenizer_path().is_some());
        assert_eq!(download.model_paths().len(), 1);
    }
}