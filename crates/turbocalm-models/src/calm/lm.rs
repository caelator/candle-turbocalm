//! CALM Language Model — Llama-style backbone for continuous vector prediction
//!
//! This implements the transformer that operates in CALM's continuous latent space.
//! It takes encoded vectors from the autoencoder and predicts next vectors autoregressively.

use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::{embedding, linear, rms_norm, Embedding, Linear, RmsNorm, VarBuilder};
use serde::{Deserialize, Serialize};

use super::autoencoder::CalmAutoencoderConfig;

/// Configuration for the CALM language model
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
}

// ── RMS Norm ─────────────────────────────────────────────────────

struct RmsNormLayer {
    inner: RmsNorm,
}

impl RmsNormLayer {
    fn new(size: usize, eps: f64, vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            inner: rms_norm(size, eps, vb)?,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        self.inner.forward(x)
    }
}

// ── Rotary Position Embeddings ───────────────────────────────────

struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    fn new(head_dim: usize, max_seq_len: usize, theta: f64, device: &Device) -> candle_core::Result<Self> {
        let half_dim = head_dim / 2;
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / (theta as f32).powf(2.0 * i as f32 / head_dim as f32))
            .collect();
        let inv_freq = Tensor::new(inv_freq, device)?;

        let positions: Vec<f32> = (0..max_seq_len).map(|p| p as f32).collect();
        let positions = Tensor::new(positions, device)?.unsqueeze(1)?;
        let freqs = positions.matmul(&inv_freq.unsqueeze(0)?)?;
        let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?;

        Ok(Self {
            cos: emb.cos()?,
            sin: emb.sin()?,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> candle_core::Result<(Tensor, Tensor)> {
        let seq_len = q.dim(1)?;
        let cos = self.cos.narrow(0, offset, seq_len)?.unsqueeze(0)?.unsqueeze(2)?;
        let sin = self.sin.narrow(0, offset, seq_len)?.unsqueeze(0)?.unsqueeze(2)?;

        let q_embed = q.broadcast_mul(&cos)?.add(&rotate_half(q)?.broadcast_mul(&sin)?)?;
        let k_embed = k.broadcast_mul(&cos)?.add(&rotate_half(k)?.broadcast_mul(&sin)?)?;
        Ok((q_embed, k_embed))
    }
}

fn rotate_half(x: &Tensor) -> candle_core::Result<Tensor> {
    let half = x.dim(D::Minus1)? / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)
}

// ── Attention ────────────────────────────────────────────────────

struct CalmAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    rotary: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    hidden_size: usize,
}

impl CalmAttention {
    fn new(config: &CalmLmConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let head_dim = config.head_dim();
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        Ok(Self {
            q_proj: linear(config.hidden_size, num_heads * head_dim, vb.pp("q_proj"))?,
            k_proj: linear(config.hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?,
            v_proj: linear(config.hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?,
            o_proj: linear(num_heads * head_dim, config.hidden_size, vb.pp("o_proj"))?,
            rotary: RotaryEmbedding::new(head_dim, config.max_position_embeddings, config.rope_theta, vb.device())?,
            num_heads,
            num_kv_heads,
            head_dim,
            hidden_size: config.hidden_size,
        })
    }

    fn forward(&self, x: &Tensor, mask: Option<&Tensor>, offset: usize) -> candle_core::Result<Tensor> {
        let (b, seq, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?.reshape((b, seq, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let k = self.k_proj.forward(x)?.reshape((b, seq, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;
        let v = self.v_proj.forward(x)?.reshape((b, seq, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;

        let (q, k) = self.rotary.apply(&q, &k, offset)?;

        // Repeat KV heads if needed (GQA)
        let k = self.repeat_kv(k)?;
        let v = self.repeat_kv(v)?;

        let scale = (self.head_dim as f64).sqrt();
        let attn = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?.affine(1.0 / scale, 0.0)?;

        let attn = match mask {
            Some(m) => attn.broadcast_add(m)?,
            None => attn,
        };

        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?;
        let out = out.transpose(1, 2)?.reshape((b, seq, self.hidden_size))?;
        self.o_proj.forward(&out)
    }

    fn repeat_kv(&self, x: Tensor) -> candle_core::Result<Tensor> {
        let rep = self.num_heads / self.num_kv_heads;
        if rep == 1 {
            return Ok(x);
        }
        let (b, h, s, d) = x.dims4()?;
        x.unsqueeze(2)?
            .expand((b, h, rep, s, d))?
            .reshape((b, h * rep, s, d))
    }
}

// ── MLP ──────────────────────────────────────────────────────────

struct CalmMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl CalmMlp {
    fn new(config: &CalmLmConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            gate_proj: linear(config.hidden_size, config.intermediate_size, vb.pp("gate_proj"))?,
            up_proj: linear(config.hidden_size, config.intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear(config.intermediate_size, config.hidden_size, vb.pp("down_proj"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let gate = self.gate_proj.forward(x)?.silu()?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

// ── Decoder Layer ────────────────────────────────────────────────

struct CalmDecoderLayer {
    self_attn: CalmAttention,
    mlp: CalmMlp,
    input_layernorm: RmsNormLayer,
    post_attention_layernorm: RmsNormLayer,
}

impl CalmDecoderLayer {
    fn new(config: &CalmLmConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            self_attn: CalmAttention::new(config, vb.pp("self_attn"))?,
            mlp: CalmMlp::new(config, vb.pp("mlp"))?,
            input_layernorm: RmsNormLayer::new(config.hidden_size, config.rms_norm_eps, vb.pp("input_layernorm"))?,
            post_attention_layernorm: RmsNormLayer::new(config.hidden_size, config.rms_norm_eps, vb.pp("post_attention_layernorm"))?,
        })
    }

    fn forward(&self, x: &Tensor, mask: Option<&Tensor>, offset: usize) -> candle_core::Result<Tensor> {
        let residual = x.clone();
        let x = self.input_layernorm.forward(x)?;
        let x = self.self_attn.forward(&x, mask, offset)?;
        let x = (residual + x)?;

        let residual = x.clone();
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        residual + x
    }
}

// ── Embed Projection ─────────────────────────────────────────────

/// Projects autoencoder latent vectors into the LM hidden space
struct EmbedProjection {
    layers: Vec<Linear>,
    norm: RmsNormLayer,
}

impl EmbedProjection {
    fn new(config: &CalmLmConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let input_size = config.patch_size * config.hidden_size;
        let layers = vec![
            linear(input_size, 2 * config.hidden_size, vb.pp("linear1"))?,
            linear(2 * config.hidden_size, config.hidden_size, vb.pp("linear2"))?,
        ];
        let norm = RmsNormLayer::new(config.hidden_size, config.rms_norm_eps, vb.pp("norm"))?;
        Ok(Self { layers, norm })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let xs = self.layers[0].forward(xs)?.silu()?;
        let xs = self.layers[1].forward(&xs)?;
        self.norm.forward(&xs)
    }
}

// ── CALM Language Model ──────────────────────────────────────────

/// The full CALM language model
pub struct CalmLanguageModel {
    config: CalmLmConfig,
    layers: Vec<CalmDecoderLayer>,
    norm: RmsNormLayer,
    embed_proj: EmbedProjection,
}

impl CalmLanguageModel {
    pub fn new(config: &CalmLmConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let mut layers = Vec::new();
        for i in 0..config.num_hidden_layers {
            layers.push(CalmDecoderLayer::new(config, vb.pp(format!("layers.{i}")))?);
        }
        Ok(Self {
            config: config.clone(),
            layers,
            norm: RmsNormLayer::new(config.hidden_size, config.rms_norm_eps, vb.pp("norm"))?,
            embed_proj: EmbedProjection::new(config, vb.pp("embed_proj"))?,
        })
    }

    /// Forward pass: latent vectors → hidden states
    pub fn forward(&self, latent_input: &Tensor, mask: Option<&Tensor>, offset: usize) -> candle_core::Result<Tensor> {
        let mut x = self.embed_proj.forward(latent_input)?;
        for layer in &self.layers {
            x = layer.forward(&x, mask, offset)?;
        }
        self.norm.forward(&x)
    }

    pub fn config(&self) -> &CalmLmConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
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
    fn test_lm_forward_shape() {
        let device = Device::Cpu;
        let config = CalmLmConfig {
            num_hidden_layers: 2,
            ..CalmLmConfig::default()
        };
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let model = CalmLanguageModel::new(&config, vb).unwrap();

        // Input: (batch=1, seq=4, patch_size*hidden=4096) — as if from autoencoder latents
        let input = Tensor::zeros((1, 4, config.patch_size * config.hidden_size), DType::F32, &device).unwrap();
        let output = model.forward(&input, None, 0).unwrap();

        assert_eq!(output.dims(), &[1, 4, config.hidden_size]);
    }

    #[test]
    fn test_rotate_half() {
        let device = Device::Cpu;
        let x = Tensor::new(&[[1.0f32, 2.0, 3.0, 4.0]], &device).unwrap();
        let rotated = rotate_half(&x).unwrap();
        let vals: Vec<f32> = rotated.flatten_all().unwrap().to_vec1().unwrap();
        // rotate_half: [-x2, x1] where x1=[1,2], x2=[3,4]
        assert_eq!(vals, vec![-3.0, -4.0, 1.0, 2.0]);
    }
}
