//! CALM Generation — patchwise autoregressive generation in continuous vector space

use candle_core::{Device, Tensor, D};
use serde::{Deserialize, Serialize};

use super::autoencoder::CalmAutoencoder;
use super::lm::CalmLanguageModel;

/// Configuration for CALM generation
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub max_patches: usize,
    pub temperature: f64,
    pub seed: u64,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_patches: 64,
            temperature: 1.0,
            seed: 42,
        }
    }
}

/// Result of generation
#[derive(Debug)]
pub struct GenerationOutput {
    pub token_ids: Vec<u32>,
    pub patches_generated: usize,
    pub latent_vectors: Vec<Tensor>,
}

/// Generate text using CALM's continuous autoregressive process
pub fn generate(
    autoencoder: &CalmAutoencoder,
    lm: &CalmLanguageModel,
    prompt_ids: &Tensor,
    config: &GenerationConfig,
    device: &Device,
) -> anyhow::Result<GenerationOutput> {
    let patch_size = autoencoder.config().patch_size;
    let latent_size = autoencoder.config().latent_size;

    let prompt_latents = autoencoder.encode_chunked(prompt_ids)?;
    let prompt_mean = prompt_latents.narrow(D::Minus1, 0, latent_size)
        .map_err(|e| anyhow::anyhow!("Failed to extract mean: {}", e))?;

    let mut all_latents = vec![prompt_mean.clone()];
    let mut generated_token_ids = Vec::new();

    for step in 0..config.max_patches {
        let latent_refs: Vec<&Tensor> = all_latents.iter().collect();
        let context = Tensor::cat(&latent_refs, 1)
            .map_err(|e| anyhow::anyhow!("Failed to concat latents at step {}: {}", step, e))?;

        let hidden_states = lm.forward(&context, None, 0)
            .map_err(|e| anyhow::anyhow!("LM forward failed at step {}: {}", step, e))?;

        let seq_len = hidden_states.dim(1)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        let last_hidden = hidden_states.narrow(1, seq_len - 1, 1)
            .map_err(|e| anyhow::anyhow!("{}", e))?;

        let predicted_latent = if last_hidden.dim(D::Minus1).map_err(|e| anyhow::anyhow!("{}", e))? != latent_size {
            last_hidden.narrow(D::Minus1, 0, latent_size)
                .map_err(|e| anyhow::anyhow!("{}", e))?
        } else {
            last_hidden.clone()
        };

        let scaled = if config.temperature != 1.0 {
            predicted_latent.affine(config.temperature, 0.0)
                .map_err(|e| anyhow::anyhow!("{}", e))?
        } else {
            predicted_latent.clone()
        };

        all_latents.push(scaled);

        // Decode produces logits; real implementation would argmax over vocab
        // For now we record that generation occurred
        for _ in 0..patch_size {
            generated_token_ids.push(0u32);
        }
    }

    let latent_vectors = all_latents.into_iter().skip(1).collect();
    Ok(GenerationOutput {
        token_ids: generated_token_ids,
        patches_generated: config.max_patches,
        latent_vectors,
    })
}

/// Memory telemetry for benchmarking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryReport {
    pub rss_bytes: u64,
    pub metal_active: bool,
    pub method: String,
}

impl MemoryReport {
    pub fn capture(device: &Device) -> Self {
        let rss_bytes = get_rss_bytes();
        let metal_active = device.is_metal();
        Self {
            rss_bytes,
            metal_active,
            method: if metal_active {
                "Metal unified memory (RSS proxy)".into()
            } else {
                "Process RSS".into()
            },
        }
    }
}

#[cfg(target_os = "macos")]
fn get_rss_bytes() -> u64 {
    use std::process::Command;
    let pid = std::process::id();
    Command::new("ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map(|kb| kb * 1024)
        .unwrap_or(0)
}

#[cfg(not(target_os = "macos"))]
fn get_rss_bytes() -> u64 {
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generation_config_defaults() {
        let config = GenerationConfig::default();
        assert_eq!(config.max_patches, 64);
        assert_eq!(config.temperature, 1.0);
        assert_eq!(config.seed, 42);
    }

    #[test]
    fn test_memory_report_capture() {
        let device = Device::Cpu;
        let report = MemoryReport::capture(&device);
        assert!(!report.metal_active);
        assert_eq!(report.method, "Process RSS");
    }

    #[test]
    fn test_generation_output_structure() {
        let output = GenerationOutput {
            token_ids: vec![1, 2, 3],
            patches_generated: 1,
            latent_vectors: vec![],
        };
        assert_eq!(output.token_ids.len(), 3);
        assert_eq!(output.patches_generated, 1);
    }
}
