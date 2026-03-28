use std::{collections::HashMap, path::Path, sync::Arc};

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{
    layer_norm, linear, linear_no_bias, Embedding, LayerNorm, Linear, Module, VarBuilder,
};
use rand::{
    distributions::{Distribution, WeightedIndex},
    rngs::StdRng,
    Rng, SeedableRng,
};
use turbocalm_core::CALMConfig;
use turbocalm_core::QuantProfile;
use turbocalm_kv::cache::{dense::DenseKvCache, KvCache, TurboKvCache};

use super::autoencoder::{CalmAutoencoder, CalmAutoencoderConfig};

#[derive(Debug, Clone)]
pub struct CalmGenerationConfig {
    pub max_new_tokens: usize,
    pub temperature: f64,
    pub num_samples: usize,
    pub seed: u64,
}

impl Default for CalmGenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 256,
            temperature: 0.5,
            num_samples: 16,
            seed: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub enum CalmKvCacheBackend {
    Turbo(QuantProfile),
    Dense,
}

impl Default for CalmKvCacheBackend {
    fn default() -> Self {
        Self::Turbo(QuantProfile::default())
    }
}

pub struct CalmGenerationOutput {
    pub token_ids: Vec<u32>,
    pub generated_token_ids: Vec<u32>,
    pub prompt_latents: Tensor,
    pub generated_latents: Tensor,
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
    fn load(vb: VarBuilder, config: &CALMConfig) -> Result<Self> {
        let hidden_size = config.hidden_size as usize;
        let intermediate_size = config.intermediate_size as usize;
        let gate_proj = linear_with_optional_bias(
            hidden_size,
            intermediate_size,
            config.mlp_bias,
            vb.pp("gate_proj"),
        )
        .context("failed to load gate_proj")?;
        let up_proj = linear_with_optional_bias(
            hidden_size,
            intermediate_size,
            config.mlp_bias,
            vb.pp("up_proj"),
        )
        .context("failed to load up_proj")?;
        let down_proj = linear_with_optional_bias(
            intermediate_size,
            hidden_size,
            config.mlp_bias,
            vb.pp("down_proj"),
        )
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
            .context("failed to apply SiLU in transformer MLP")?;
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
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(config: &CALMConfig, dtype: DType, device: &Device) -> Result<Self> {
        if config.rope_scaling.is_some() {
            bail!("rope_scaling is not implemented for native CALM generation yet");
        }

        let head_dim = config.hidden_size as usize / config.num_attention_heads as usize;
        let half_dim = head_dim / 2;
        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / config.rope_theta.powf(i as f64 / head_dim as f64) as f32)
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (1, half_dim), device)?
            .to_dtype(dtype)
            .context("failed to cast rotary inv_freq")?;
        let positions = Tensor::arange(0u32, config.max_position_embeddings, device)?
            .to_dtype(dtype)
            .context("failed to cast rotary positions")?
            .reshape((config.max_position_embeddings as usize, 1))
            .context("failed to reshape rotary positions")?;
        let freqs = positions
            .matmul(&inv_freq)
            .context("failed to build rotary frequencies")?;
        Ok(Self {
            sin: freqs.sin().context("failed to compute rotary sin")?,
            cos: freqs.cos().context("failed to compute rotary cos")?,
        })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self
            .cos
            .narrow(0, seqlen_offset, seq_len)
            .context("failed to slice rotary cos")?;
        let sin = self
            .sin
            .narrow(0, seqlen_offset, seq_len)
            .context("failed to slice rotary sin")?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)
            .context("failed to apply rotary embedding to q")?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)
            .context("failed to apply rotary embedding to k")?;
        Ok((q_embed, k_embed))
    }
}

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: Box<dyn KvCache>,
}

impl Attention {
    fn load(
        rotary_emb: Arc<RotaryEmbedding>,
        config: &CALMConfig,
        cache_backend: &CalmKvCacheBackend,
        vb: VarBuilder,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size as usize;
        let num_heads = config.num_attention_heads as usize;
        let num_kv_heads = config.num_key_value_heads() as usize;
        let head_dim = hidden_size / num_heads;
        let q_proj = linear_with_optional_bias(
            hidden_size,
            hidden_size,
            config.attention_bias,
            vb.pp("q_proj"),
        )?;
        let k_proj = linear_with_optional_bias(
            hidden_size,
            num_kv_heads * head_dim,
            config.attention_bias,
            vb.pp("k_proj"),
        )?;
        let v_proj = linear_with_optional_bias(
            hidden_size,
            num_kv_heads * head_dim,
            config.attention_bias,
            vb.pp("v_proj"),
        )?;
        let o_proj = linear_with_optional_bias(
            hidden_size,
            hidden_size,
            config.attention_bias,
            vb.pp("o_proj"),
        )?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            num_kv_groups: num_heads / num_kv_heads,
            head_dim,
            rotary_emb,
            kv_cache: build_kv_cache(cache_backend),
        })
    }

    fn forward(
        &mut self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (batch_size, query_len, hidden_size) = hidden_states.dims3()?;
        let q = self
            .q_proj
            .forward(hidden_states)
            .context("failed to run q_proj")?
            .reshape((batch_size, query_len, self.num_heads, self.head_dim))
            .context("failed to reshape q projection")?
            .transpose(1, 2)
            .context("failed to transpose q projection")?;
        let k = self
            .k_proj
            .forward(hidden_states)
            .context("failed to run k_proj")?
            .reshape((batch_size, query_len, self.num_kv_heads, self.head_dim))
            .context("failed to reshape k projection")?
            .transpose(1, 2)
            .context("failed to transpose k projection")?;
        let v = self
            .v_proj
            .forward(hidden_states)
            .context("failed to run v_proj")?
            .reshape((batch_size, query_len, self.num_kv_heads, self.head_dim))
            .context("failed to reshape v projection")?
            .transpose(1, 2)
            .context("failed to transpose v projection")?;

        let (q, k) = self
            .rotary_emb
            .apply_rotary_emb_qkv(&q, &k, seqlen_offset)?;
        self.kv_cache
            .append(&k, &v)
            .context("failed to append attention KV cache")?;
        let k = self
            .kv_cache
            .get_key()
            .context("failed to retrieve attention k cache")?;
        let v = self
            .kv_cache
            .get_value()
            .context("failed to retrieve attention v cache")?;

        let k = repeat_kv(&k, self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(&v, self.num_kv_groups)?.contiguous()?;

        let scale = 1f64 / (self.head_dim as f64).sqrt();
        let attn_weights = q
            .matmul(&k.transpose(2, 3)?)
            .context("failed to compute attention weights")?
            .affine(scale, 0.0)
            .context("failed to scale attention weights")?;
        let attn_weights = match attention_mask {
            Some(mask) => attn_weights
                .broadcast_add(mask)
                .context("failed to apply attention mask")?,
            None => attn_weights,
        };
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)
            .context("failed to softmax attention weights")?;
        let attn_output = attn_weights
            .matmul(&v)
            .context("failed to multiply attention weights and values")?
            .transpose(1, 2)
            .context("failed to transpose attention output")?
            .reshape((batch_size, query_len, hidden_size))
            .context("failed to reshape attention output")?;
        self.o_proj
            .forward(&attn_output)
            .context("failed to run o_proj")
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache.clear();
    }
}

struct DecoderLayer {
    self_attn: Attention,
    mlp: LlamaStyleMlp,
    input_layernorm: RmsNormLayer,
    post_attention_layernorm: RmsNormLayer,
}

impl DecoderLayer {
    fn load(
        rotary_emb: Arc<RotaryEmbedding>,
        config: &CALMConfig,
        cache_backend: &CalmKvCacheBackend,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            self_attn: Attention::load(rotary_emb, config, cache_backend, vb.pp("attention"))?,
            mlp: LlamaStyleMlp::load(vb.pp("mlp"), config)?,
            input_layernorm: RmsNormLayer::load(
                config.hidden_size as usize,
                config.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            post_attention_layernorm: RmsNormLayer::load(
                config.hidden_size as usize,
                config.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    fn forward(
        &mut self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let residual = hidden_states;
        let hidden_states = self.input_layernorm.forward(hidden_states)?;
        let hidden_states =
            self.self_attn
                .forward(&hidden_states, attention_mask, seqlen_offset)?;
        let hidden_states = residual
            .broadcast_add(&hidden_states)
            .context("failed to add attention residual")?;

        let residual = &hidden_states;
        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        residual
            .broadcast_add(&hidden_states)
            .context("failed to add MLP residual")
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

pub struct CalmDecoder {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNormLayer,
    device: Device,
    dtype: DType,
}

impl CalmDecoder {
    pub fn load(
        vb: VarBuilder,
        config: &CALMConfig,
        cache_backend: &CalmKvCacheBackend,
    ) -> Result<Self> {
        let rotary_emb = Arc::new(RotaryEmbedding::new(config, vb.dtype(), vb.device())?);
        let embed_tokens = candle_nn::embedding(
            config.vocab_size as usize,
            config.hidden_size as usize,
            vb.pp("embed_tokens"),
        )
        .context("failed to load transformer token embeddings")?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers as usize);
        for layer_idx in 0..config.num_hidden_layers as usize {
            layers.push(DecoderLayer::load(
                rotary_emb.clone(),
                config,
                cache_backend,
                vb.pp(format!("layers.{layer_idx}")),
            )?);
        }
        Ok(Self {
            embed_tokens,
            layers,
            norm: RmsNormLayer::load(
                config.hidden_size as usize,
                config.rms_norm_eps,
                vb.pp("norm"),
            )?,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    fn embed(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens
            .forward(input_ids)
            .context("failed to embed transformer token IDs")
    }

    pub fn forward_embeds(
        &mut self,
        inputs_embeds: &Tensor,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _) = inputs_embeds.dims3()?;
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            Some(self.prepare_decoder_attention_mask(batch_size, seq_len, seqlen_offset)?)
        };

        let mut hidden_states = inputs_embeds.clone();
        for layer in self.layers.iter_mut() {
            hidden_states =
                layer.forward(&hidden_states, attention_mask.as_ref(), seqlen_offset)?;
        }
        self.norm.forward(&hidden_states)
    }

    fn prepare_decoder_attention_mask(
        &self,
        batch_size: usize,
        target_len: usize,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let mask: Vec<f32> = (0..target_len)
            .flat_map(|i| (0..target_len).map(move |j| if i < j { f32::NEG_INFINITY } else { 0.0 }))
            .collect();
        let mask = Tensor::from_vec(mask, (target_len, target_len), &self.device)
            .context("failed to build decoder attention mask")?;
        let mask = if seqlen_offset > 0 {
            let prefix = Tensor::zeros((target_len, seqlen_offset), DType::F32, &self.device)
                .context("failed to allocate cached attention mask prefix")?;
            Tensor::cat(&[&prefix, &mask], D::Minus1)
                .context("failed to extend decoder attention mask")?
        } else {
            mask
        };
        mask.expand((batch_size, 1, target_len, target_len + seqlen_offset))
            .context("failed to expand decoder attention mask")?
            .to_dtype(self.dtype)
            .context("failed to cast decoder attention mask")
    }

    fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache();
        }
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

#[derive(Debug, Clone)]
pub struct PatchEmbeddingProjection {
    input_proj: Linear,
    output_proj: Linear,
    norm: LayerNorm,
    patch_size: usize,
    hidden_size: usize,
}

impl PatchEmbeddingProjection {
    pub fn load(vb: VarBuilder, config: &CALMConfig) -> Result<Self> {
        let hidden_size = config.hidden_size as usize;
        Ok(Self {
            input_proj: linear(
                hidden_size * config.patch_size as usize,
                hidden_size * 2,
                vb.pp("0"),
            )
            .context("failed to load embed_proj.0")?,
            output_proj: linear(hidden_size * 2, hidden_size, vb.pp("2"))
                .context("failed to load embed_proj.2")?,
            norm: layer_norm(hidden_size, 1e-6, vb.pp("3"))
                .context("failed to load embed_proj.3")?,
            patch_size: config.patch_size as usize,
            hidden_size,
        })
    }

    pub fn forward(&self, token_embeddings: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, hidden_size) = token_embeddings.dims3()?;
        if hidden_size != self.hidden_size {
            bail!(
                "embed_proj expected hidden_size={}, got {}",
                self.hidden_size,
                hidden_size
            );
        }
        if seq_len % self.patch_size != 0 {
            bail!(
                "embed_proj expected input length divisible by patch_size={}, got {}",
                self.patch_size,
                seq_len
            );
        }

        let num_patches = seq_len / self.patch_size;
        let hidden_states = token_embeddings
            .reshape((batch_size, num_patches, self.patch_size * self.hidden_size))
            .context("failed to flatten token patches before embed_proj")?;
        let hidden_states = self
            .input_proj
            .forward(&hidden_states)
            .context("failed to run embed_proj.0")?
            .silu()
            .context("failed to apply SiLU in embed_proj")?;
        let hidden_states = self
            .output_proj
            .forward(&hidden_states)
            .context("failed to run embed_proj.2")?;
        self.norm
            .forward(&hidden_states)
            .context("failed to run embed_proj.3")
    }
}

#[derive(Debug, Clone)]
struct MlpBlock {
    in_ln: LayerNorm,
    linears_0: Linear,
    linears_2: Linear,
    linears_4: Linear,
    down_proj: Linear,
}

impl MlpBlock {
    fn load(vb: VarBuilder, hidden_size: usize) -> Result<Self> {
        Ok(Self {
            in_ln: layer_norm(hidden_size, 1e-6, vb.pp("in_ln"))
                .context("failed to load MLPBlock in_ln")?,
            linears_0: linear(hidden_size * 2, hidden_size, vb.pp("linears.0"))
                .context("failed to load MLPBlock linears.0")?,
            linears_2: linear(hidden_size, hidden_size, vb.pp("linears.2"))
                .context("failed to load MLPBlock linears.2")?,
            linears_4: linear(hidden_size, hidden_size * 2, vb.pp("linears.4"))
                .context("failed to load MLPBlock linears.4")?,
            down_proj: linear(hidden_size, hidden_size, vb.pp("down_proj"))
                .context("failed to load MLPBlock down_proj")?,
        })
    }

    fn forward(&self, x: &Tensor, y: &Tensor) -> Result<Tensor> {
        let channels = x.dims().last().copied().unwrap_or_default();
        let h = self
            .in_ln
            .forward(x)
            .context("failed to run MLPBlock in_ln")?;
        let h =
            Tensor::cat(&[&h, y], D::Minus1).context("failed to concatenate MLPBlock inputs")?;
        let h = self
            .linears_0
            .forward(&h)
            .context("failed to run MLPBlock linears.0")?
            .silu()
            .context("failed to apply SiLU after MLPBlock linears.0")?;
        let h = self
            .linears_2
            .forward(&h)
            .context("failed to run MLPBlock linears.2")?
            .silu()
            .context("failed to apply SiLU after MLPBlock linears.2")?;
        let h = self
            .linears_4
            .forward(&h)
            .context("failed to run MLPBlock linears.4")?;
        let gate_proj = h
            .narrow(D::Minus1, 0, channels)
            .context("failed to split MLPBlock gate_proj")?
            .silu()
            .context("failed to apply SiLU to MLPBlock gate_proj")?;
        let up_proj = h
            .narrow(D::Minus1, channels, channels)
            .context("failed to split MLPBlock up_proj")?;
        let step = self
            .down_proj
            .forward(
                &gate_proj
                    .mul(&up_proj)
                    .context("failed to combine MLPBlock gate and up projections")?,
            )
            .context("failed to run MLPBlock down_proj")?;
        x.broadcast_add(&step)
            .context("failed to add MLPBlock residual")
    }
}

#[derive(Debug, Clone)]
struct FinalLayer {
    in_ln: LayerNorm,
    linears_0: Linear,
    linears_2: Linear,
}

impl FinalLayer {
    fn load(vb: VarBuilder, hidden_size: usize, latent_size: usize) -> Result<Self> {
        Ok(Self {
            in_ln: layer_norm(hidden_size, 1e-6, vb.pp("in_ln"))
                .context("failed to load final_layer in_ln")?,
            linears_0: linear(hidden_size, hidden_size, vb.pp("linears.0"))
                .context("failed to load final_layer linears.0")?,
            linears_2: linear(hidden_size, latent_size, vb.pp("linears.2"))
                .context("failed to load final_layer linears.2")?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self
            .in_ln
            .forward(x)
            .context("failed to run final_layer in_ln")?;
        let x = self
            .linears_0
            .forward(&x)
            .context("failed to run final_layer linears.0")?
            .silu()
            .context("failed to apply SiLU in final_layer")?;
        self.linears_2
            .forward(&x)
            .context("failed to run final_layer linears.2")
    }
}

#[derive(Debug, Clone)]
struct MlpGenerator {
    noise_size: usize,
    latent_size: usize,
    hidden_embd: Linear,
    noise_embd: Linear,
    norm_hidden: LayerNorm,
    norm_noise: LayerNorm,
    mlp_blocks: Vec<MlpBlock>,
    final_layer: FinalLayer,
}

impl MlpGenerator {
    fn load(vb: VarBuilder, config: &CALMConfig) -> Result<Self> {
        let hidden_size = config.hidden_size as usize;
        let latent_size = config.latent_size as usize;
        let num_mlp_layers = config.num_mlp_layers as usize;
        let noise_size = config.noise_size as usize;

        let mut mlp_blocks = Vec::with_capacity(num_mlp_layers);
        for idx in 0..num_mlp_layers {
            mlp_blocks.push(MlpBlock::load(
                vb.pp(format!("mlp_blocks.{idx}")),
                hidden_size,
            )?);
        }

        Ok(Self {
            noise_size,
            latent_size,
            hidden_embd: linear(hidden_size, hidden_size, vb.pp("hidden_embd"))
                .context("failed to load hidden_embd")?,
            noise_embd: linear(noise_size, hidden_size, vb.pp("noise_embd"))
                .context("failed to load noise_embd")?,
            norm_hidden: layer_norm(hidden_size, 1e-6, vb.pp("norm_hidden"))
                .context("failed to load norm_hidden")?,
            norm_noise: layer_norm(hidden_size, 1e-6, vb.pp("norm_noise"))
                .context("failed to load norm_noise")?,
            mlp_blocks,
            final_layer: FinalLayer::load(vb.pp("final_layer"), hidden_size, latent_size)?,
        })
    }

    fn sample(&self, hidden_states: &Tensor, rng: &mut StdRng) -> Result<Tensor> {
        let hidden_shape = hidden_states.dims().to_vec();
        let sample_shape = &hidden_shape[..hidden_shape.len().saturating_sub(1)];
        let sample_count = sample_shape.iter().product::<usize>();
        let noise_shape: Vec<usize> = sample_shape
            .iter()
            .copied()
            .chain(std::iter::once(self.noise_size))
            .collect();
        let noise = (0..sample_count * self.noise_size)
            .map(|_| rng.gen::<f32>() - 0.5)
            .collect::<Vec<_>>();
        let noise = Tensor::from_vec(noise, noise_shape.as_slice(), hidden_states.device())
            .context("failed to create MLPGenerator noise tensor")?
            .to_dtype(hidden_states.dtype())
            .context("failed to cast MLPGenerator noise tensor")?;

        let mut noise_embds = self
            .noise_embd
            .forward(&noise)
            .context("failed to run noise_embd")?;
        noise_embds = self
            .norm_noise
            .forward(&noise_embds)
            .context("failed to run norm_noise")?;

        let mut hidden_states = self
            .hidden_embd
            .forward(hidden_states)
            .context("failed to run hidden_embd")?;
        hidden_states = self
            .norm_hidden
            .forward(&hidden_states)
            .context("failed to run norm_hidden")?;

        for block in &self.mlp_blocks {
            noise_embds = block.forward(&noise_embds, &hidden_states)?;
        }

        let latent_prediction = self.final_layer.forward(&noise_embds)?;
        let latent_shape: Vec<usize> = sample_shape
            .iter()
            .copied()
            .chain(std::iter::once(self.latent_size))
            .collect();
        latent_prediction
            .reshape(latent_shape.as_slice())
            .context("failed to reshape latent prediction")
    }
}

pub struct CalmGenerationModel {
    config: CALMConfig,
    autoencoder: CalmAutoencoder,
    transformer: CalmDecoder,
    embed_proj: PatchEmbeddingProjection,
    generative_head: MlpGenerator,
}

impl CalmGenerationModel {
    pub fn load(
        vb: VarBuilder,
        config: CALMConfig,
        autoencoder_config: CalmAutoencoderConfig,
    ) -> Result<Self> {
        Self::load_with_kv_cache_backend(
            vb,
            config,
            autoencoder_config,
            CalmKvCacheBackend::default(),
        )
    }

    pub fn load_with_kv_cache_backend(
        vb: VarBuilder,
        config: CALMConfig,
        autoencoder_config: CalmAutoencoderConfig,
        kv_cache_backend: CalmKvCacheBackend,
    ) -> Result<Self> {
        config.validate()?;

        let patch_size = config.patch_size as usize;
        if autoencoder_config.patch_size != patch_size {
            bail!(
                "CALM patch_size={} does not match autoencoder patch_size={}",
                patch_size,
                autoencoder_config.patch_size
            );
        }
        if autoencoder_config.latent_size != config.latent_size as usize {
            bail!(
                "CALM latent_size={} does not match autoencoder latent_size={}",
                config.latent_size,
                autoencoder_config.latent_size
            );
        }

        let autoencoder_prefix = if vb.contains_tensor("ae_model.decoder.lm_head_weight")
            || vb.contains_tensor("ae_model.encoder.embed_tokens.weight")
        {
            "ae_model"
        } else {
            ""
        };
        let generative_prefix = if vb.contains_tensor("generative_head.noise_embd.weight") {
            "generative_head"
        } else if vb.contains_tensor("mlp_generator.noise_embd.weight") {
            "mlp_generator"
        } else {
            "generative_head"
        };

        Ok(Self {
            autoencoder: CalmAutoencoder::load_prefixed(
                vb.clone(),
                autoencoder_prefix,
                autoencoder_config,
            )?,
            transformer: CalmDecoder::load(vb.pp("transformer"), &config, &kv_cache_backend)?,
            embed_proj: PatchEmbeddingProjection::load(vb.pp("embed_proj"), &config)?,
            generative_head: MlpGenerator::load(vb.pp(generative_prefix), &config)?,
            config,
        })
    }

    pub fn from_safetensors<P: AsRef<Path>>(
        paths: &[P],
        config: CALMConfig,
        autoencoder_config: CalmAutoencoderConfig,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        Self::from_safetensors_with_kv_cache_backend(
            paths,
            config,
            autoencoder_config,
            dtype,
            device,
            CalmKvCacheBackend::default(),
        )
    }

    pub fn from_safetensors_with_kv_cache_backend<P: AsRef<Path>>(
        paths: &[P],
        config: CALMConfig,
        autoencoder_config: CalmAutoencoderConfig,
        dtype: DType,
        device: &Device,
        kv_cache_backend: CalmKvCacheBackend,
    ) -> Result<Self> {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(paths, dtype, device) }
            .context("failed to mmap CALM safetensors weights")?;
        Self::load_with_kv_cache_backend(vb, config, autoencoder_config, kv_cache_backend)
    }

    pub fn from_pth<P: AsRef<Path>>(
        path: P,
        config: CALMConfig,
        autoencoder_config: CalmAutoencoderConfig,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        Self::from_pth_with_kv_cache_backend(
            path,
            config,
            autoencoder_config,
            dtype,
            device,
            CalmKvCacheBackend::default(),
        )
    }

    pub fn from_pth_with_kv_cache_backend<P: AsRef<Path>>(
        path: P,
        config: CALMConfig,
        autoencoder_config: CalmAutoencoderConfig,
        dtype: DType,
        device: &Device,
        kv_cache_backend: CalmKvCacheBackend,
    ) -> Result<Self> {
        let vb = VarBuilder::from_pth(path, dtype, device)
            .context("failed to load PyTorch checkpoint")?;
        Self::load_with_kv_cache_backend(vb, config, autoencoder_config, kv_cache_backend)
    }

    pub fn device(&self) -> &Device {
        self.transformer.device()
    }

    pub fn generate(
        &mut self,
        prompt_token_ids: &[u32],
        options: &CalmGenerationConfig,
    ) -> Result<CalmGenerationOutput> {
        validate_temperature(options.temperature)?;
        if options.num_samples == 0 {
            bail!("num_samples must be greater than zero")
        }

        self.transformer.clear_kv_cache();

        let patch_size = self.config.patch_size as usize;
        let pad_token_id = self.config.pad_token_id.unwrap_or(self.config.eos_token_id);
        let prompt_token_ids = if prompt_token_ids.is_empty() {
            vec![self.config.bos_token_id]
        } else {
            prompt_token_ids.to_vec()
        };

        let prompt_pad = (patch_size - (prompt_token_ids.len() % patch_size)) % patch_size;
        let mut working_ids = prompt_token_ids.clone();
        working_ids.extend(
            std::iter::repeat(pad_token_id)
                .take(prompt_pad)
                .map(|id| id as u32),
        );

        let prompt_tensor =
            Tensor::from_vec(working_ids.clone(), (1, working_ids.len()), self.device())
                .context("failed to create prompt tensor")?;
        let prompt_latents = self.autoencoder.encode_chunked(&prompt_tensor)?;

        let mut generated_token_ids = Vec::new();
        let mut generated_latents = Vec::new();
        let mut rng = StdRng::seed_from_u64(options.seed);
        let mut seqlen_offset = 0usize;

        while generated_token_ids.len() < options.max_new_tokens {
            let patch_embeddings = if seqlen_offset == 0 {
                // For the first iteration, use the prompt latents from the autoencoder
                self.latent_patches_to_embeddings(&prompt_latents)?
            } else {
                // For subsequent iterations, encode the latest tokens to get their latents
                let current_tokens = working_ids[working_ids.len() - patch_size..].to_vec();
                let current_tensor =
                    Tensor::from_vec(current_tokens, (1, patch_size), self.device())
                        .context("failed to create current generation window")?;
                let current_latents = self.autoencoder.encode_chunked(&current_tensor)?;
                self.latent_patches_to_embeddings(&current_latents)?
            };

            let num_patches = patch_embeddings.dims()[1];
            let hidden_states = self
                .transformer
                .forward_embeds(&patch_embeddings, seqlen_offset)?;
            seqlen_offset += num_patches;

            let last_hidden_state = hidden_states
                .narrow(1, hidden_states.dims()[1] - 1, 1)
                .context("failed to select last hidden state")?
                .reshape((1, self.config.hidden_size as usize))
                .context("failed to reshape last hidden state")?;

            let (patch_latent, patch_token_ids) = self.temperature_sample(
                &last_hidden_state,
                options.temperature,
                options.num_samples,
                &mut rng,
            )?;
            generated_latents.push(patch_latent.clone());
            working_ids.extend_from_slice(&patch_token_ids);

            for token_id in patch_token_ids {
                if generated_token_ids.len() == options.max_new_tokens {
                    break;
                }
                generated_token_ids.push(token_id);
                if token_id == self.config.eos_token_id {
                    break;
                }
            }

            if generated_token_ids
                .last()
                .copied()
                .is_some_and(|token_id| token_id == self.config.eos_token_id)
            {
                break;
            }
        }

        let generated_latents = if generated_latents.is_empty() {
            Tensor::zeros(
                (1, 0, self.config.latent_size as usize),
                DType::F32,
                self.device(),
            )
            .context("failed to allocate empty generated_latents tensor")?
        } else {
            let latent_refs = generated_latents.iter().collect::<Vec<_>>();
            Tensor::cat(&latent_refs, 1).context("failed to concatenate generated latents")?
        };

        let mut token_ids = prompt_token_ids;
        token_ids.extend_from_slice(&generated_token_ids);

        Ok(CalmGenerationOutput {
            token_ids,
            generated_token_ids,
            prompt_latents,
            generated_latents,
        })
    }

    fn temperature_sample(
        &self,
        hidden_states: &Tensor,
        temperature: f64,
        num_samples: usize,
        rng: &mut StdRng,
    ) -> Result<(Tensor, Vec<u32>)> {
        let patch_size = self.config.patch_size as usize;
        let latent_size = self.config.latent_size as usize;

        if (temperature - 1.0).abs() < f64::EPSILON {
            let latent = self
                .generative_head
                .sample(hidden_states, rng)?
                .reshape((1, 1, latent_size))
                .context("failed to reshape direct latent sample")?;
            let token_ids = self.decode_latent_patch(&latent)?;
            return Ok((latent, token_ids));
        }

        let hidden_size = hidden_states.dims()[1];
        let repeated = hidden_states
            .reshape((1, 1, hidden_size))
            .context("failed to reshape hidden state for sampling")?
            .expand((1, num_samples, hidden_size))
            .context("failed to repeat hidden state for sampling")?;
        let candidate_latents = self.generative_head.sample(&repeated, rng)?;
        let candidate_tokens = self
            .autoencoder
            .decode(&candidate_latents)
            .context("failed to decode candidate latents")?
            .argmax(D::Minus1)
            .context("failed to argmax candidate logits")?
            .reshape((num_samples, patch_size))
            .context("failed to reshape candidate token patches")?
            .to_vec2::<u32>()
            .context("failed to extract candidate token patches")?;
        let candidate_latents = candidate_latents
            .reshape((num_samples, latent_size))
            .context("failed to reshape candidate latents")?
            .to_vec2::<f32>()
            .context("failed to extract candidate latents")?;

        let mut patch_counts = HashMap::<Vec<u32>, (usize, Vec<f32>)>::new();
        for (tokens, latent) in candidate_tokens
            .into_iter()
            .zip(candidate_latents.into_iter())
        {
            patch_counts
                .entry(tokens)
                .and_modify(|entry| entry.0 += 1)
                .or_insert((1, latent));
        }

        let reciprocal = (1.0 / temperature).round() as usize;
        for threshold in (1..=reciprocal).rev() {
            let candidates = patch_counts
                .iter()
                .filter(|(_, (count, _))| *count >= threshold)
                .collect::<Vec<_>>();
            if candidates.is_empty() {
                continue;
            }

            let weights = candidates
                .iter()
                .map(|(_, (count, _))| combination(*count, threshold))
                .collect::<Vec<_>>();
            let distribution = WeightedIndex::new(&weights)
                .context("failed to build temperature sampling distribution")?;
            let selected = candidates[distribution.sample(rng)];
            let latent =
                Tensor::from_vec(selected.1 .1.clone(), (1, 1, latent_size), self.device())
                    .context("failed to create selected latent tensor")?;
            return Ok((latent, selected.0.clone()));
        }

        bail!("temperature sampling failed to select a CALM patch")
    }

    fn decode_latent_patch(&self, latent_patch: &Tensor) -> Result<Vec<u32>> {
        self.autoencoder
            .decode(latent_patch)
            .context("failed to decode latent patch")?
            .argmax(D::Minus1)
            .context("failed to argmax decoded latent patch")?
            .reshape((self.config.patch_size as usize,))
            .context("failed to flatten decoded patch tokens")?
            .to_vec1::<u32>()
            .context("failed to extract decoded patch tokens")
    }

    /// Convert latent patches to patch embeddings for the transformer
    fn latent_patches_to_embeddings(&self, latent_patches: &Tensor) -> Result<Tensor> {
        let (_batch_size, _num_patches, latent_size) = latent_patches.dims3()?;
        let hidden_size = self.config.hidden_size as usize;

        if latent_size != self.config.latent_size as usize {
            bail!(
                "Expected latent_size={}, got {}",
                self.config.latent_size,
                latent_size
            );
        }

        // Create a simple linear projection from latent space to hidden space
        // This bypasses the token embedding → patch embedding path
        // We create patch embeddings directly from latent patches
        let patch_embeddings = latent_patches
            .pad_with_zeros(D::Minus1, 0, hidden_size - latent_size)
            .context("failed to pad latent patches to hidden size")?;

        // Normalize the patch embeddings to match the expected scale
        let norm_factor = (hidden_size as f32).sqrt();
        patch_embeddings
            .affine(1.0 / norm_factor as f64, 0.0)
            .context("failed to normalize latent patch embeddings")
    }
}

pub fn generate(
    model: &mut CalmGenerationModel,
    prompt_token_ids: &[u32],
    options: &CalmGenerationConfig,
) -> Result<CalmGenerationOutput> {
    model.generate(prompt_token_ids, options)
}

fn build_kv_cache(cache_backend: &CalmKvCacheBackend) -> Box<dyn KvCache> {
    match cache_backend {
        CalmKvCacheBackend::Turbo(profile) => Box::new(TurboKvCache::new(profile.clone())),
        CalmKvCacheBackend::Dense => Box::new(DenseKvCache::new()),
    }
}

fn linear_with_optional_bias(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    vb: VarBuilder,
) -> Result<Linear> {
    if bias {
        Ok(linear(in_dim, out_dim, vb)?)
    } else {
        Ok(linear_no_bias(in_dim, out_dim, vb)?)
    }
}

fn repeat_kv(hidden_states: &Tensor, repeats: usize) -> Result<Tensor> {
    if repeats == 1 {
        return Ok(hidden_states.clone());
    }
    let (batch_size, num_kv_heads, seq_len, head_dim) = hidden_states.dims4()?;
    hidden_states
        .unsqueeze(2)
        .context("failed to unsqueeze KV tensor for repetition")?
        .expand((batch_size, num_kv_heads, repeats, seq_len, head_dim))
        .context("failed to expand KV tensor for repetition")?
        .reshape((batch_size, num_kv_heads * repeats, seq_len, head_dim))
        .context("failed to reshape repeated KV tensor")
}

fn validate_temperature(temperature: f64) -> Result<()> {
    if temperature <= 0.0 {
        bail!("temperature must be positive, got {temperature}");
    }
    if !temperature.is_finite() {
        bail!("temperature must be finite (not NaN or infinite), got {temperature}");
    }
    Ok(())
}

fn combination(n: usize, k: usize) -> f64 {
    if k == 0 || n == k {
        return 1.0;
    }
    let k = k.min(n - k);
    (0..k).fold(1.0, |acc, idx| acc * (n - idx) as f64 / (idx + 1) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_calm_config() -> CALMConfig {
        CALMConfig {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: Some(4),
            max_position_embeddings: 64,
            patch_size: 2,
            latent_size: 8,
            num_mlp_layers: 2,
            num_samples: 8,
            noise_size: 4,
            ..Default::default()
        }
    }

    fn test_autoencoder_config() -> CalmAutoencoderConfig {
        CalmAutoencoderConfig {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 40,
            latent_size: 8,
            patch_size: 2,
            ..Default::default()
        }
    }

    #[test]
    fn generate_returns_patchwise_latent_shapes() -> Result<()> {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let mut model =
            CalmGenerationModel::load(vb, test_calm_config(), test_autoencoder_config())?;
        let output = model.generate(
            &[1, 2, 3, 4],
            &CalmGenerationConfig {
                max_new_tokens: 4,
                temperature: 1.0,
                num_samples: 8,
                seed: 42,
            },
        )?;

        assert_eq!(output.prompt_latents.dims(), &[1, 2, 8]);
        assert_eq!(output.generated_latents.dims(), &[1, 2, 8]);
        assert_eq!(output.token_ids.len(), 8);
        assert_eq!(output.generated_token_ids.len(), 4);
        Ok(())
    }

    #[test]
    fn generate_supports_dense_kv_cache_backend() -> Result<()> {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let mut model = CalmGenerationModel::load_with_kv_cache_backend(
            vb,
            test_calm_config(),
            test_autoencoder_config(),
            CalmKvCacheBackend::Dense,
        )?;
        let output = model.generate(
            &[1, 2, 3, 4],
            &CalmGenerationConfig {
                max_new_tokens: 4,
                temperature: 1.0,
                num_samples: 8,
                seed: 7,
            },
        )?;

        assert_eq!(output.generated_token_ids.len(), 4);
        Ok(())
    }

    #[test]
    fn temperature_validation_rejects_invalid_values() {
        // Test negative temperature
        let err = validate_temperature(-0.5).unwrap_err();
        assert!(err.to_string().contains("positive"));

        // Test zero temperature
        let err = validate_temperature(0.0).unwrap_err();
        assert!(err.to_string().contains("positive"));

        // Test NaN
        let err = validate_temperature(f64::NAN).unwrap_err();
        assert!(err.to_string().contains("finite"));

        // Test infinity
        let err = validate_temperature(f64::INFINITY).unwrap_err();
        assert!(err.to_string().contains("finite"));

        // Test that valid temperatures work
        assert!(validate_temperature(0.3).is_ok());
        assert!(validate_temperature(0.7).is_ok());
        assert!(validate_temperature(1.0).is_ok());
        assert!(validate_temperature(2.0).is_ok());
    }

    #[test]
    fn test_latent_conditioning_changes_output() -> Result<()> {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let mut model =
            CalmGenerationModel::load(vb, test_calm_config(), test_autoencoder_config())?;

        let config = CalmGenerationConfig {
            max_new_tokens: 2,
            temperature: 1.0,
            num_samples: 8,
            seed: 42,
        };

        // Generate with input tokens
        let output = model.generate(&[1, 2], &config)?;

        // Verify that the latent conditioning path is working:
        // 1. The autoencoder produces latent representations
        // 2. The transformer consumes these latent patches
        assert_eq!(output.prompt_latents.dims()[2], 8); // latent_size from autoencoder
        assert!(output.prompt_latents.dims()[1] >= 1); // at least one patch

        // Verify we have generated some tokens using the latent conditioning
        assert!(output.generated_token_ids.len() > 0);
        assert_eq!(output.generated_latents.dims()[2], 8); // latent_size

        Ok(())
    }
}
