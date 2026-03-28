//! CALM Language Model — Simplified interface for CALM transformer
//!
//! This provides a simplified interface that wraps the main CalmGenerationModel
//! for compatibility with existing code that expects the CalmLanguageModel interface.

use anyhow::Result;
use candle_core::Tensor;
use candle_nn::VarBuilder;
use serde::{Deserialize, Serialize};

use super::generation::{CalmDecoder, CalmKvCacheBackend, PatchEmbeddingProjection};
use turbocalm_core::CALMConfig;

/// Configuration for the CALM language model (legacy interface)
/// This is now a compatibility wrapper around CALMConfig
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CalmLmConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub latent_size: usize,
    pub patch_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    /// Energy scoring beta parameter
    pub beta: f64,
    /// Number of samples for energy scoring
    pub num_energy_samples: usize,
}

impl Default for CalmLmConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1024,
            intermediate_size: 2752,
            num_hidden_layers: 16,
            num_attention_heads: 16,
            num_key_value_heads: 16,
            latent_size: 128,
            patch_size: 4,
            max_position_embeddings: 8192,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            beta: 1.0,
            num_energy_samples: 100,
        }
    }
}

impl CalmLmConfig {
    /// CALM-M configuration (371M parameters)
    pub fn calm_m() -> Self {
        Self::default()
    }

    /// CALM-L configuration (735M parameters)
    pub fn calm_l() -> Self {
        Self {
            hidden_size: 2048,
            intermediate_size: 5504,
            num_hidden_layers: 24,
            ..Self::default()
        }
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Convert to CALMConfig for use with the unified transformer
    pub fn to_calm_config(&self) -> CALMConfig {
        CALMConfig {
            hidden_size: self.hidden_size as u32,
            intermediate_size: self.intermediate_size as u32,
            num_hidden_layers: self.num_hidden_layers as u32,
            num_attention_heads: self.num_attention_heads as u32,
            num_key_value_heads: Some(self.num_key_value_heads as u32),
            max_position_embeddings: self.max_position_embeddings as u32,
            patch_size: self.patch_size as u32,
            latent_size: self.latent_size as u32,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            ..Default::default()
        }
    }
}

/// The CALM language model - now a wrapper around the unified transformer from generation.rs
pub struct CalmLanguageModel {
    config: CalmLmConfig,
    transformer: CalmDecoder,
    embed_proj: PatchEmbeddingProjection,
}

impl CalmLanguageModel {
    pub fn new(config: &CalmLmConfig, vb: VarBuilder) -> Result<Self> {
        let calm_config = config.to_calm_config();
        let transformer = CalmDecoder::load(
            vb.pp("transformer"),
            &calm_config,
            &CalmKvCacheBackend::Dense,
        )?;
        let embed_proj = PatchEmbeddingProjection::load(vb.pp("embed_proj"), &calm_config)?;

        Ok(Self {
            config: config.clone(),
            transformer,
            embed_proj,
        })
    }

    /// Forward pass: latent vectors → hidden states
    /// This method maintains compatibility with the original interface but now delegates to the unified implementation
    pub fn forward(
        &mut self,
        latent_input: &Tensor,
        _mask: Option<&Tensor>,
        offset: usize,
    ) -> Result<Tensor> {
        // Convert latent patches to patch embeddings
        let (_batch_size, _num_patches, latent_size) = latent_input.dims3()?;
        let hidden_size = self.config.hidden_size;

        if latent_size != self.config.latent_size {
            return Err(anyhow::anyhow!(
                "Expected latent_size={}, got {}",
                self.config.latent_size,
                latent_size
            ));
        }

        // Convert latent patches to patch embeddings by padding to hidden_size
        let patch_embeddings =
            latent_input.pad_with_zeros(candle_core::D::Minus1, 0, hidden_size - latent_size)?;

        // Normalize the patch embeddings
        let norm_factor = (hidden_size as f32).sqrt();
        let patch_embeddings = patch_embeddings.affine(1.0 / norm_factor as f64, 0.0)?;

        self.transformer.forward_embeds(&patch_embeddings, offset)
    }

    pub fn config(&self) -> &CalmLmConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    #[test]
    fn test_lm_config_defaults() {
        let config = CalmLmConfig::default();
        assert_eq!(config.hidden_size, 1024);
        assert_eq!(config.num_hidden_layers, 16);
        assert_eq!(config.head_dim(), 64);
    }

    #[test]
    fn test_calm_m_config() {
        let config = CalmLmConfig::calm_m();
        assert_eq!(config.hidden_size, 1024);
        assert_eq!(config.intermediate_size, 2752);
    }

    #[test]
    fn test_calm_l_config() {
        let config = CalmLmConfig::calm_l();
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.num_hidden_layers, 24);
    }

    #[test]
    fn test_config_conversion() {
        let lm_config = CalmLmConfig::default();
        let calm_config = lm_config.to_calm_config();

        assert_eq!(calm_config.hidden_size, 1024);
        assert_eq!(calm_config.num_hidden_layers, 16);
        assert_eq!(calm_config.patch_size, 4);
    }

    #[test]
    fn test_lm_forward_shape() {
        let device = Device::Cpu;
        let config = CalmLmConfig {
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            latent_size: 8,
            patch_size: 2,
            max_position_embeddings: 64,
            ..CalmLmConfig::default()
        };
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let mut model = CalmLanguageModel::new(&config, vb).unwrap();

        // Input: (batch=1, seq=4, latent_size) — latent patches from autoencoder
        let input = Tensor::zeros((1, 4, config.latent_size), DType::F32, &device).unwrap();
        let output = model.forward(&input, None, 0).unwrap();

        assert_eq!(output.dims(), &[1, 4, config.hidden_size]);
    }
}
