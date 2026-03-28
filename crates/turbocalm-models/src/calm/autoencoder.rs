use std::{fs, path::Path};

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{linear, linear_no_bias, Embedding, Linear, Module, VarBuilder};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct CalmAutoencoderConfig {
    pub ae_dropout: f64,
    pub kl_clamp: f64,
    pub kl_weight: f64,
    pub patch_size: usize,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_encoder_layers: usize,
    pub num_decoder_layers: usize,
    pub latent_size: usize,
    pub hidden_act: String,
    pub max_position_embeddings: usize,
    pub initializer_range: f64,
    pub rms_norm_eps: f64,
    pub pad_token_id: Option<u32>,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub pretraining_tp: usize,
    pub tie_word_embeddings: bool,
    pub mlp_bias: bool,
}

impl Default for CalmAutoencoderConfig {
    fn default() -> Self {
        Self {
            ae_dropout: 0.15,
            kl_clamp: 0.5,
            kl_weight: 1e-3,
            patch_size: 4,
            vocab_size: 32_000,
            hidden_size: 512,
            intermediate_size: 1_280,
            num_encoder_layers: 2,
            num_decoder_layers: 2,
            latent_size: 128,
            hidden_act: "silu".to_string(),
            max_position_embeddings: 2_048,
            initializer_range: 0.02,
            rms_norm_eps: 1e-6,
            pad_token_id: None,
            bos_token_id: 1,
            eos_token_id: 2,
            pretraining_tp: 1,
            tie_word_embeddings: false,
            mlp_bias: false,
        }
    }
}

impl CalmAutoencoderConfig {
    pub fn from_json_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let raw = fs::read_to_string(path).with_context(|| {
            format!("failed to read autoencoder config from {}", path.display())
        })?;
        serde_json::from_str(&raw)
            .with_context(|| format!("failed to parse autoencoder config at {}", path.display()))
    }

    fn validate(&self) -> Result<()> {
        if self.patch_size == 0 {
            bail!("patch_size must be greater than zero")
        }
        if self.hidden_size == 0 {
            bail!("hidden_size must be greater than zero")
        }
        if self.intermediate_size == 0 {
            bail!("intermediate_size must be greater than zero")
        }
        if self.latent_size == 0 {
            bail!("latent_size must be greater than zero")
        }
        if self.num_encoder_layers == 0 || self.num_encoder_layers % 2 != 0 {
            bail!("num_encoder_layers must be a positive even number")
        }
        if self.num_decoder_layers == 0 || self.num_decoder_layers % 2 != 0 {
            bail!("num_decoder_layers must be a positive even number")
        }
        if self.hidden_act != "silu" {
            bail!(
                "unsupported hidden_act {:?}; only \"silu\" matches the CALM autoencoder",
                self.hidden_act
            )
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct RmsNormLayer {
    weight: Tensor,
    eps: f64,
}

impl RmsNormLayer {
    fn load(hidden_size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb
            .get(hidden_size, "weight")
            .context("failed to load RMSNorm weight")?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        candle_nn::ops::rms_norm(hidden_states, &self.weight, self.eps as f32)
            .context("failed to run RMSNorm")
    }
}

#[derive(Debug, Clone)]
struct LlamaStyleMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl LlamaStyleMlp {
    fn load(vb: VarBuilder, config: &CalmAutoencoderConfig) -> Result<Self> {
        let gate_proj = if config.mlp_bias {
            linear(
                config.hidden_size,
                config.intermediate_size,
                vb.pp("gate_proj"),
            )
        } else {
            linear_no_bias(
                config.hidden_size,
                config.intermediate_size,
                vb.pp("gate_proj"),
            )
        }
        .context("failed to load gate_proj")?;

        let up_proj = if config.mlp_bias {
            linear(
                config.hidden_size,
                config.intermediate_size,
                vb.pp("up_proj"),
            )
        } else {
            linear_no_bias(
                config.hidden_size,
                config.intermediate_size,
                vb.pp("up_proj"),
            )
        }
        .context("failed to load up_proj")?;

        let down_proj = if config.mlp_bias {
            linear(
                config.intermediate_size,
                config.hidden_size,
                vb.pp("down_proj"),
            )
        } else {
            linear_no_bias(
                config.intermediate_size,
                config.hidden_size,
                vb.pp("down_proj"),
            )
        }
        .context("failed to load down_proj")?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let gate = self
            .gate_proj
            .forward(hidden_states)
            .context("failed to run gate_proj")?
            .silu()
            .context("failed to apply SiLU in encoder/decoder MLP")?;
        let up = self
            .up_proj
            .forward(hidden_states)
            .context("failed to run up_proj")?;
        let hidden_states = gate
            .mul(&up)
            .context("failed to combine gated MLP activations")?;
        self.down_proj
            .forward(&hidden_states)
            .context("failed to run down_proj")
    }
}

#[derive(Debug, Clone)]
struct AutoencoderLayer {
    mlp: LlamaStyleMlp,
    layernorm: RmsNormLayer,
}

impl AutoencoderLayer {
    fn load(vb: VarBuilder, config: &CalmAutoencoderConfig) -> Result<Self> {
        Ok(Self {
            mlp: LlamaStyleMlp::load(vb.pp("mlp"), config)?,
            layernorm: RmsNormLayer::load(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp("layernorm"),
            )?,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let residual = hidden_states;
        let hidden_states = self.layernorm.forward(hidden_states)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        residual
            .broadcast_add(&hidden_states)
            .context("failed to add autoencoder residual")
    }
}

#[derive(Debug, Clone)]
pub struct CalmAutoencoderEncoder {
    patch_size: usize,
    hidden_size: usize,
    latent_size: usize,
    num_stage_layers: usize,
    embed_tokens: Embedding,
    encoder_layers: Vec<AutoencoderLayer>,
    hidden_to_latent: Linear,
    squeeze_layer: Linear,
    norm: RmsNormLayer,
}

impl CalmAutoencoderEncoder {
    fn load(vb: VarBuilder, config: &CalmAutoencoderConfig, embedding_weight: Tensor) -> Result<Self> {
        let embed_tokens = Embedding::new(embedding_weight, config.hidden_size);

        let encoder_layers = (0..config.num_encoder_layers)
            .map(|idx| AutoencoderLayer::load(vb.pp(format!("encoder_layers.{idx}")), config))
            .collect::<Result<Vec<_>>>()?;

        let hidden_to_latent = linear(
            config.hidden_size,
            config.latent_size * 2,
            vb.pp("hidden_to_latent"),
        )
        .context("failed to load encoder hidden_to_latent")?;

        let squeeze_layer = linear(
            config.patch_size * config.hidden_size,
            config.hidden_size,
            vb.pp("squeeze_layer"),
        )
        .context("failed to load encoder squeeze_layer")?;

        let norm = RmsNormLayer::load(config.hidden_size, config.rms_norm_eps, vb.pp("norm"))?;

        Ok(Self {
            patch_size: config.patch_size,
            hidden_size: config.hidden_size,
            latent_size: config.latent_size,
            num_stage_layers: config.num_encoder_layers / 2,
            embed_tokens,
            encoder_layers,
            hidden_to_latent,
            squeeze_layer,
            norm,
        })
    }

    fn embedding_weight(&self) -> Tensor {
        self.embed_tokens.embeddings().clone()
    }

    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_length) = input_ids
            .dims2()
            .context("encoder expects token IDs with shape [batch, seq]")?;
        if seq_length % self.patch_size != 0 {
            bail!(
                "encoder expects input length divisible by patch_size={}, got {}",
                self.patch_size,
                seq_length
            )
        }

        let num_patches = seq_length / self.patch_size;
        let input_ids = input_ids
            .reshape((batch_size * num_patches, self.patch_size))
            .context("failed to reshape token IDs into patches")?;

        let mut hidden_states = self
            .embed_tokens
            .forward(&input_ids)
            .context("failed to embed input token IDs")?;

        for stage in 0..2 {
            for layer_offset in 0..self.num_stage_layers {
                let layer_idx = stage * self.num_stage_layers + layer_offset;
                hidden_states = self.encoder_layers[layer_idx].forward(&hidden_states)?;
            }

            if stage == 0 {
                hidden_states = hidden_states
                    .reshape((
                        batch_size * num_patches,
                        1,
                        self.patch_size * self.hidden_size,
                    ))
                    .context("failed to flatten each patch before squeeze_layer")?;
                hidden_states = self
                    .squeeze_layer
                    .forward(&hidden_states)
                    .context("failed to run squeeze_layer")?;
            }
        }

        let hidden_states = self.norm.forward(&hidden_states)?;
        let latent_states = self
            .hidden_to_latent
            .forward(&hidden_states)
            .context("failed to project encoder hidden states into latent statistics")?;

        latent_states
            .reshape((batch_size, num_patches, self.latent_size * 2))
            .context("failed to reshape encoder latent statistics back to [batch, patches, latent]")
    }
}

#[derive(Debug, Clone)]
pub struct CalmAutoencoderDecoder {
    patch_size: usize,
    hidden_size: usize,
    latent_size: usize,
    num_stage_layers: usize,
    latent_to_hidden: Linear,
    decoder_layers: Vec<AutoencoderLayer>,
    expand_layer: Linear,
    norm: RmsNormLayer,
    lm_head: Linear,
}

impl CalmAutoencoderDecoder {
    fn load(vb: VarBuilder, config: &CalmAutoencoderConfig, lm_head: Linear) -> Result<Self> {
        let decoder_layers = (0..config.num_decoder_layers)
            .map(|idx| AutoencoderLayer::load(vb.pp(format!("decoder_layers.{idx}")), config))
            .collect::<Result<Vec<_>>>()?;

        let latent_to_hidden = linear(
            config.latent_size,
            config.hidden_size,
            vb.pp("latent_to_hidden"),
        )
        .context("failed to load decoder latent_to_hidden")?;

        let expand_layer = linear(
            config.hidden_size,
            config.patch_size * config.hidden_size,
            vb.pp("expand_layer"),
        )
        .context("failed to load decoder expand_layer")?;

        let norm = RmsNormLayer::load(config.hidden_size, config.rms_norm_eps, vb.pp("norm"))?;

        Ok(Self {
            patch_size: config.patch_size,
            hidden_size: config.hidden_size,
            latent_size: config.latent_size,
            num_stage_layers: config.num_decoder_layers / 2,
            latent_to_hidden,
            decoder_layers,
            expand_layer,
            norm,
            lm_head,
        })
    }

    pub fn forward(&self, latent_states: &Tensor) -> Result<Tensor> {
        let (batch_size, num_patches, latent_size) = latent_states.dims3().context(
            "decoder expects latent states with shape [batch, num_patches, latent_size]",
        )?;
        if latent_size != self.latent_size {
            bail!(
                "decoder expected latent_size={}, got {}",
                self.latent_size,
                latent_size
            )
        }

        let mut hidden_states = self
            .latent_to_hidden
            .forward(latent_states)
            .context("failed to project latent states into decoder hidden states")?;

        for stage in 0..2 {
            for layer_offset in 0..self.num_stage_layers {
                let layer_idx = stage * self.num_stage_layers + layer_offset;
                hidden_states = self.decoder_layers[layer_idx].forward(&hidden_states)?;
            }

            if stage == 0 {
                hidden_states = self
                    .expand_layer
                    .forward(&hidden_states)
                    .context("failed to run decoder expand_layer")?;
                hidden_states = hidden_states
                    .reshape((batch_size, num_patches * self.patch_size, self.hidden_size))
                    .context("failed to expand patches back into token positions")?;
            }
        }

        let hidden_states = self.norm.forward(&hidden_states)?;
        self.lm_head
            .forward(&hidden_states)
            .context("failed to project decoder hidden states into vocabulary logits")
    }
}

#[derive(Debug, Clone)]
pub struct CalmAutoencoder {
    config: CalmAutoencoderConfig,
    encoder: CalmAutoencoderEncoder,
    decoder: CalmAutoencoderDecoder,
}

impl CalmAutoencoder {
    pub fn load(vb: VarBuilder, config: CalmAutoencoderConfig) -> Result<Self> {
        Self::load_prefixed(vb, "", config)
    }

    pub fn load_prefixed(vb: VarBuilder, prefix: &str, config: CalmAutoencoderConfig) -> Result<Self> {
        config.validate()?;

        let vb = if prefix.is_empty() { vb } else { vb.pp(prefix) };
        let embedding_weight = Self::load_embedding_weight(&vb, &config)?;
        let encoder = CalmAutoencoderEncoder::load(vb.pp("encoder"), &config, embedding_weight.clone())?;
        let lm_head = Linear::new(embedding_weight, None);
        let decoder = CalmAutoencoderDecoder::load(vb.pp("decoder"), &config, lm_head)?;

        Ok(Self {
            config,
            encoder,
            decoder,
        })
    }

    fn load_embedding_weight(vb: &VarBuilder, config: &CalmAutoencoderConfig) -> Result<Tensor> {
        let shape = (config.vocab_size, config.hidden_size);
        if vb.contains_tensor("encoder.embed_tokens.weight") {
            vb.pp("encoder")
                .get(shape, "embed_tokens.weight")
                .context("failed to load encoder.embed_tokens.weight")
        } else if vb.contains_tensor("decoder.lm_head_weight") {
            vb.pp("decoder")
                .get(shape, "lm_head_weight")
                .context("failed to load decoder.lm_head_weight")
        } else if vb.contains_tensor("decoder.lm_head.weight") {
            vb.pp("decoder")
                .get(shape, "lm_head.weight")
                .context("failed to load decoder.lm_head.weight")
        } else {
            bail!("autoencoder checkpoint is missing shared embedding weights")
        }
    }

    pub fn from_safetensors<P: AsRef<Path>>(
        path: P,
        config: CalmAutoencoderConfig,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let path = path.as_ref();
        let data = fs::read(path)
            .with_context(|| format!("failed to read safetensors file {}", path.display()))?;
        let vb = VarBuilder::from_buffered_safetensors(data, dtype, device)
            .with_context(|| format!("failed to open safetensors weights {}", path.display()))?;
        Self::load_prefixed(vb, "", config)
    }

    pub fn config(&self) -> &CalmAutoencoderConfig {
        &self.config
    }

    fn pad_input_ids(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_length) = input_ids
            .dims2()
            .context("encode expects token IDs with shape [batch, seq]")?;
        let remainder = seq_length % self.config.patch_size;
        if remainder == 0 {
            return Ok(input_ids.clone());
        }

        let pad_len = self.config.patch_size - remainder;
        let pad_value = self.config.pad_token_id.unwrap_or_default();
        let padding = Tensor::full(pad_value, (batch_size, pad_len), input_ids.device())
            .context("failed to create padding tensor for token IDs")?
            .to_dtype(input_ids.dtype())
            .context("failed to cast padding tensor to input token dtype")?;

        Tensor::cat(&[input_ids, &padding], 1)
            .context("failed to append patch padding to token IDs")
    }

    fn encode_latent_stats(&self, input_ids: &Tensor) -> Result<Tensor> {
        let padded = self.pad_input_ids(input_ids)?;
        self.encoder.forward(&padded)
    }

    pub fn encode_chunked(&self, input_ids: &Tensor) -> Result<Tensor> {
        let latent_stats = self.encode_latent_stats(input_ids)?;
        latent_stats
            .narrow(D::Minus1, 0, self.config.latent_size)
            .context("failed to split encoder latent stats into mean/log_std")
    }

    pub fn encode_pooled(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.encode_chunked(input_ids)?
            .mean(1)
            .context("failed to average chunk embeddings across patches")
    }

    pub fn decode(&self, latent_states: &Tensor) -> Result<Tensor> {
        self.decoder.forward(latent_states)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> CalmAutoencoderConfig {
        CalmAutoencoderConfig {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 40,
            latent_size: 8,
            ..Default::default()
        }
    }

    fn test_model(config: CalmAutoencoderConfig) -> Result<CalmAutoencoder> {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        CalmAutoencoder::load(vb, config)
    }

    #[test]
    fn config_defaults_match_python_reference() {
        let config = CalmAutoencoderConfig::default();
        assert_eq!(config.ae_dropout, 0.15);
        assert_eq!(config.kl_clamp, 0.5);
        assert_eq!(config.kl_weight, 1e-3);
        assert_eq!(config.patch_size, 4);
        assert_eq!(config.vocab_size, 32_000);
        assert_eq!(config.hidden_size, 512);
        assert_eq!(config.intermediate_size, 1_280);
        assert_eq!(config.num_encoder_layers, 2);
        assert_eq!(config.num_decoder_layers, 2);
        assert_eq!(config.latent_size, 128);
        assert_eq!(config.hidden_act, "silu");
        assert_eq!(config.max_position_embeddings, 2_048);
        assert_eq!(config.rms_norm_eps, 1e-6);
        assert_eq!(config.bos_token_id, 1);
        assert_eq!(config.eos_token_id, 2);
        assert_eq!(config.pretraining_tp, 1);
        assert!(!config.tie_word_embeddings);
        assert!(!config.mlp_bias);
    }

    #[test]
    fn encode_shapes_follow_patch_padding_and_latent_split() -> Result<()> {
        let device = Device::Cpu;
        let config = test_config();
        let model = test_model(config.clone())?;
        let input_ids = Tensor::from_vec(vec![1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10], (2, 5), &device)?;

        let latent_stats = model.encode_latent_stats(&input_ids)?;
        assert_eq!(latent_stats.dims(), &[2, 2, config.latent_size * 2]);

        let chunked = model.encode_chunked(&input_ids)?;
        assert_eq!(chunked.dims(), &[2, 2, config.latent_size]);

        let pooled = model.encode_pooled(&input_ids)?;
        assert_eq!(pooled.dims(), &[2, config.latent_size]);

        Ok(())
    }

    #[test]
    fn decode_shape_matches_patch_expansion() -> Result<()> {
        let device = Device::Cpu;
        let config = test_config();
        let model = test_model(config.clone())?;
        let latent_states = Tensor::zeros((2, 3, config.latent_size), DType::F32, &device)?;

        let logits = model.decode(&latent_states)?;
        assert_eq!(logits.dims(), &[2, 12, config.vocab_size]);

        Ok(())
    }
}
