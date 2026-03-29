//! turbocalm-triumvirate - Triumvirate integration adapters for TurboCALM
//!
//! This crate provides adapter interfaces for integrating TurboCALM models with the Triumvirate framework.
//! It wraps the CalmAutoencoder and CalmLanguageModel to provide standard embedding and scoring interfaces.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use candle_core::{DType, Tensor};
use candle_nn::VarBuilder;
use serde::{Deserialize, Serialize};
use turbocalm_checkpoint::CheckpointDownloader;
use turbocalm_core::auto_device;
use turbocalm_models::{CalmAutoencoder, CalmAutoencoderConfig, CalmLanguageModel, CalmLmConfig};

/// Embedding engine configuration for Triumvirate.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct EmbeddingEngineConfig {
    pub prefer_trained: bool,
}

impl Default for EmbeddingEngineConfig {
    fn default() -> Self {
        Self {
            prefer_trained: true,
        }
    }
}

/// Embedding engine interface for Triumvirate framework
///
/// Wraps CalmAutoencoder to provide a standard embedding interface that can be used
/// by Triumvirate for text-to-embedding transformations.
pub struct EmbeddingEngine {
    autoencoder: Option<CalmAutoencoder>,
    config_path: String,
    autoencoder_config: Option<CalmAutoencoderConfig>,
    engine_config: EmbeddingEngineConfig,
}

impl EmbeddingEngine {
    /// Create a new embedding engine with the specified config path or model ID.
    pub fn new<P: AsRef<Path>>(config_path: P) -> Self {
        Self::with_config(config_path, EmbeddingEngineConfig::default())
    }

    /// Create a new embedding engine with explicit settings.
    pub fn with_config<P: AsRef<Path>>(config_path: P, engine_config: EmbeddingEngineConfig) -> Self {
        Self {
            autoencoder: None,
            config_path: config_path.as_ref().to_string_lossy().to_string(),
            autoencoder_config: None,
            engine_config,
        }
    }

    /// Load the underlying CalmAutoencoder model.
    pub fn load_model(&mut self) -> Result<()> {
        let device = auto_device()?;
        let autoencoder_config = self.resolve_autoencoder_config()?;

        let autoencoder = match self.preferred_trained_checkpoint() {
            Some(path) => CalmAutoencoder::from_safetensors(
                &path,
                autoencoder_config.clone(),
                DType::F32,
                &device,
            )
            .with_context(|| format!("failed to load trained checkpoint {}", path.display()))?,
            None => self.load_fallback_autoencoder(&autoencoder_config, &device)?,
        };

        self.autoencoder = Some(autoencoder);
        self.autoencoder_config = Some(autoencoder_config);
        Ok(())
    }

    /// Generate embeddings for input text tokens.
    pub fn embed(&self, input_ids: &Tensor) -> Result<Tensor> {
        match &self.autoencoder {
            Some(autoencoder) => autoencoder.encode_chunked(input_ids),
            None => anyhow::bail!("Model not loaded - call load_model() first"),
        }
    }

    /// Get embedding dimension.
    pub fn embedding_dim(&self) -> Option<usize> {
        self.autoencoder_config.as_ref().map(|config| config.latent_size)
    }

    fn resolve_autoencoder_config(&self) -> Result<CalmAutoencoderConfig> {
        if self.engine_config.prefer_trained {
            if let Some(config) = load_trained_autoencoder_config()? {
                return Ok(config);
            }
        }

        let config_path = Path::new(&self.config_path);
        if config_path.exists() {
            return CalmAutoencoderConfig::from_json_file(config_path)
                .with_context(|| format!("failed to read {}", config_path.display()));
        }

        let downloader = CheckpointDownloader::new()?;
        let checkpoint = downloader.download_calm_checkpoint(&self.config_path)?;
        Ok(autoencoder_config_from_calm(checkpoint.calm_config()))
    }

    fn preferred_trained_checkpoint(&self) -> Option<PathBuf> {
        if !self.engine_config.prefer_trained {
            return None;
        }
        trained_checkpoint_path().filter(|path| path.exists())
    }

    fn load_fallback_autoencoder(
        &self,
        autoencoder_config: &CalmAutoencoderConfig,
        device: &candle_core::Device,
    ) -> Result<CalmAutoencoder> {
        let config_path = Path::new(&self.config_path);
        if config_path.exists() {
            return zeroed_autoencoder(autoencoder_config, device);
        }

        let downloader = CheckpointDownloader::new()?;
        let checkpoint = downloader.download_calm_checkpoint(&self.config_path)?;
        match autoencoder_from_model_paths(checkpoint.model_paths(), autoencoder_config, device) {
            Ok(autoencoder) => Ok(autoencoder),
            Err(error) => {
                eprintln!(
                    "warning: failed to load downloaded autoencoder weights for {}: {error:#}; falling back to zeroed weights",
                    self.config_path
                );
                zeroed_autoencoder(autoencoder_config, device)
            }
        }
    }
}

/// Energy scoring interface for Triumvirate framework
///
/// Wraps CalmLanguageModel to provide a standard scoring interface that can be used
/// by Triumvirate for energy-based modeling and scoring tasks.
pub struct EnergyScorer {
    language_model: Option<CalmLanguageModel>,
    config_path: String,
    lm_config: Option<CalmLmConfig>,
}

impl EnergyScorer {
    /// Create a new energy scorer with the specified config path or model ID
    ///
    /// # Arguments
    /// * `config_path` - Path to model configuration file or HuggingFace model ID
    ///
    /// # Returns
    /// A new EnergyScorer instance (model loading is deferred)
    pub fn new<P: AsRef<Path>>(config_path: P) -> Self {
        Self {
            language_model: None,
            config_path: config_path.as_ref().to_string_lossy().to_string(),
            lm_config: None,
        }
    }

    /// Load the underlying CalmLanguageModel
    ///
    /// This is a stub implementation - in a full implementation this would:
    /// 1. Load the model configuration from config_path
    /// 2. Download model weights if needed
    /// 3. Initialize the CalmLanguageModel
    pub fn load_model(&mut self) -> Result<()> {
        let downloader = CheckpointDownloader::new()?;
        let checkpoint = downloader.download_calm_checkpoint(&self.config_path)?;

        let calm_config = checkpoint.calm_config();
        let lm_config = CalmLmConfig {
            hidden_size: calm_config.hidden_size as usize,
            intermediate_size: calm_config.intermediate_size as usize,
            num_hidden_layers: calm_config.num_hidden_layers as usize,
            num_attention_heads: calm_config.num_attention_heads as usize,
            num_key_value_heads: calm_config.num_key_value_heads() as usize,
            latent_size: calm_config.latent_size as usize,
            patch_size: calm_config.patch_size as usize,
            max_position_embeddings: calm_config.max_position_embeddings as usize,
            rms_norm_eps: calm_config.rms_norm_eps,
            rope_theta: calm_config.rope_theta,
            beta: calm_config.beta,
            ..Default::default()
        };

        let device = auto_device()?;
        let var_builder = VarBuilder::zeros(DType::F32, &device);
        let language_model = CalmLanguageModel::new(&lm_config, var_builder)?;

        self.language_model = Some(language_model);
        self.lm_config = Some(lm_config);

        Ok(())
    }

    /// Compute energy scores for input embeddings
    ///
    /// # Arguments
    /// * `embeddings` - Input embeddings as tensor
    ///
    /// # Returns
    /// Tensor containing energy scores
    pub fn score(&mut self, embeddings: &Tensor) -> Result<Tensor> {
        match &mut self.language_model {
            Some(lm) => {
                let lm_output = lm.forward(embeddings, None, 0)?;
                let energy_tensor = lm_output.sum_keepdim(candle_core::D::Minus1)?;
                Ok(energy_tensor)
            }
            None => anyhow::bail!("Model not loaded - call load_model() first"),
        }
    }

    /// Get the latent dimension expected by the energy scorer
    pub fn latent_dim(&self) -> Option<usize> {
        self.lm_config.as_ref().map(|config| config.latent_size)
    }
}

fn autoencoder_config_from_calm(calm_config: turbocalm_core::CALMConfig) -> CalmAutoencoderConfig {
    CalmAutoencoderConfig {
        vocab_size: calm_config.vocab_size as usize,
        hidden_size: calm_config.hidden_size as usize,
        intermediate_size: calm_config.intermediate_size as usize,
        latent_size: calm_config.latent_size as usize,
        patch_size: calm_config.patch_size as usize,
        max_position_embeddings: calm_config.max_position_embeddings as usize,
        rms_norm_eps: calm_config.rms_norm_eps,
        hidden_act: calm_config.hidden_act,
        ..Default::default()
    }
}

fn autoencoder_from_model_paths(
    paths: &[PathBuf],
    config: &CalmAutoencoderConfig,
    device: &candle_core::Device,
) -> Result<CalmAutoencoder> {
    if paths.is_empty() {
        anyhow::bail!("no model weights were downloaded")
    }

    if paths.len() == 1 {
        return CalmAutoencoder::from_safetensors(&paths[0], config.clone(), DType::F32, device)
            .with_context(|| format!("failed to load {}", paths[0].display()));
    }

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(paths, DType::F32, device) }
        .context("failed to mmap downloaded safetensors shards")?;
    CalmAutoencoder::load(vb, config.clone()).context("failed to load autoencoder from sharded weights")
}

fn zeroed_autoencoder(
    config: &CalmAutoencoderConfig,
    device: &candle_core::Device,
) -> Result<CalmAutoencoder> {
    CalmAutoencoder::load(VarBuilder::zeros(DType::F32, device), config.clone())
        .context("failed to create zeroed autoencoder")
}

fn trained_checkpoint_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    Some(
        PathBuf::from(home)
            .join(".turbocalm")
            .join("trained")
            .join("latest.safetensors"),
    )
}

fn trained_config_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    Some(
        PathBuf::from(home)
            .join(".turbocalm")
            .join("trained")
            .join("latest-config.json"),
    )
}

fn load_trained_autoencoder_config() -> Result<Option<CalmAutoencoderConfig>> {
    let Some(path) = trained_config_path() else {
        return Ok(None);
    };
    if !path.exists() {
        return Ok(None);
    }

    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let config = serde_json::from_str(&raw)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    Ok(Some(config))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_engine_creation() {
        let engine = EmbeddingEngine::new("test/config/path");
        assert_eq!(engine.config_path, "test/config/path");
        assert!(engine.autoencoder.is_none());
        assert!(engine.engine_config.prefer_trained);
    }

    #[test]
    fn test_embedding_engine_can_disable_trained_preference() {
        let engine = EmbeddingEngine::with_config(
            "test/config/path",
            EmbeddingEngineConfig {
                prefer_trained: false,
            },
        );
        assert!(!engine.engine_config.prefer_trained);
    }

    #[test]
    fn test_energy_scorer_creation() {
        let scorer = EnergyScorer::new("test/config/path");
        assert_eq!(scorer.config_path, "test/config/path");
        assert!(scorer.language_model.is_none());
    }

    #[test]
    fn test_embedding_engine_requires_model_loading() {
        let engine = EmbeddingEngine::new("test/config/path");
        assert!(engine.embedding_dim().is_none());
    }

    #[test]
    fn test_energy_scorer_requires_model_loading() {
        let scorer = EnergyScorer::new("test/config/path");
        assert!(scorer.latent_dim().is_none());
    }
}
