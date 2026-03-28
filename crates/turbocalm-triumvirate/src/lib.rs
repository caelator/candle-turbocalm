//! turbocalm-triumvirate - Triumvirate integration adapters for TurboCALM
//!
//! This crate provides adapter interfaces for integrating TurboCALM models with the Triumvirate framework.
//! It wraps the CalmAutoencoder and CalmLanguageModel to provide standard embedding and scoring interfaces.

use anyhow::Result;
use candle_core::{DType, Tensor};
use candle_nn::VarBuilder;
use std::path::Path;
use turbocalm_models::{CalmAutoencoder, CalmAutoencoderConfig, CalmLanguageModel, CalmLmConfig};
use turbocalm_core::{CALMConfig, auto_device, hub::convenience};
use turbocalm_checkpoint::CheckpointDownloader;

/// Embedding engine interface for Triumvirate framework
///
/// Wraps CalmAutoencoder to provide a standard embedding interface that can be used
/// by Triumvirate for text-to-embedding transformations.
pub struct EmbeddingEngine {
    autoencoder: Option<CalmAutoencoder>,
    config_path: String,
    autoencoder_config: Option<CalmAutoencoderConfig>,
}

impl EmbeddingEngine {
    /// Create a new embedding engine with the specified config path or model ID
    ///
    /// # Arguments
    /// * `config_path` - Path to model configuration file or HuggingFace model ID
    ///
    /// # Returns
    /// A new EmbeddingEngine instance (model loading is deferred)
    pub fn new<P: AsRef<Path>>(config_path: P) -> Self {
        Self {
            autoencoder: None,
            config_path: config_path.as_ref().to_string_lossy().to_string(),
            autoencoder_config: None,
        }
    }

    /// Load the underlying CalmAutoencoder model
    ///
    /// This is a stub implementation - in a full implementation this would:
    /// 1. Load the model configuration from config_path
    /// 2. Download model weights if needed
    /// 3. Initialize the CalmAutoencoder
    pub fn load_model(&mut self) -> Result<()> {
        // Download checkpoint and load autoencoder config
        let downloader = CheckpointDownloader::new()?;
        let checkpoint = downloader.download_calm_checkpoint(&self.config_path)?;

        // Load CALM config and convert to autoencoder config
        let calm_config = checkpoint.calm_config();
        let autoencoder_config = CalmAutoencoderConfig {
            vocab_size: calm_config.vocab_size as usize,
            hidden_size: calm_config.hidden_size as usize,
            latent_size: calm_config.latent_size as usize,
            patch_size: calm_config.patch_size as usize,
            ..Default::default()
        };

        // Initialize device and create autoencoder with zeroed weights
        // Note: In a full implementation, this would load actual weights
        let device = auto_device()?;
        let var_builder = VarBuilder::zeros(DType::F32, &device);

        let autoencoder = CalmAutoencoder::load(var_builder, autoencoder_config.clone())?;

        self.autoencoder = Some(autoencoder);
        self.autoencoder_config = Some(autoencoder_config);

        Ok(())
    }

    /// Generate embeddings for input text tokens
    ///
    /// # Arguments
    /// * `input_ids` - Tokenized input text as tensor of token IDs
    ///
    /// # Returns
    /// Tensor containing embeddings for the input
    pub fn embed(&self, input_ids: &Tensor) -> Result<Tensor> {
        match &self.autoencoder {
            Some(autoencoder) => {
                // Tokenize text and run autoencoder encode to return latent tensor
                // Note: input_ids should already be tokenized, so we just encode
                autoencoder.encode_chunked(input_ids)
            }
            None => anyhow::bail!("Model not loaded - call load_model() first"),
        }
    }

    /// Get embedding dimension
    pub fn embedding_dim(&self) -> Option<usize> {
        // Return the config's latent_size
        self.autoencoder_config.as_ref().map(|config| config.latent_size)
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
        // Download checkpoint and load LM config
        let downloader = CheckpointDownloader::new()?;
        let checkpoint = downloader.download_calm_checkpoint(&self.config_path)?;

        // Load CALM config and convert to LM config
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

        // Initialize device and create language model with zeroed weights
        // Note: In a full implementation, this would load actual weights
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
                // Tokenize, encode, run LM forward, return energy score
                // Run the forward pass through the language model
                let lm_output = lm.forward(embeddings, None, 0)?;

                // Convert output to energy score (simple norm as placeholder)
                let energy_tensor = lm_output.sum_keepdim(candle_core::D::Minus1)?;
                Ok(energy_tensor)
            }
            None => anyhow::bail!("Model not loaded - call load_model() first"),
        }
    }

    /// Get the latent dimension expected by the energy scorer
    pub fn latent_dim(&self) -> Option<usize> {
        // Return config's latent_size
        self.lm_config.as_ref().map(|config| config.latent_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_engine_creation() {
        let engine = EmbeddingEngine::new("test/config/path");
        assert_eq!(engine.config_path, "test/config/path");
        assert!(engine.autoencoder.is_none());
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
        // Should fail since model is not loaded
        assert!(engine.embedding_dim().is_none());
    }

    #[test]
    fn test_energy_scorer_requires_model_loading() {
        let scorer = EnergyScorer::new("test/config/path");
        // Should fail since model is not loaded
        assert!(scorer.latent_dim().is_none());
    }
}
