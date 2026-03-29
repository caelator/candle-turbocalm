use anyhow::Result;
use candle_core::Tensor;

pub struct PolarQuantizer {
    pub bit_width: u8,
    pub scale_mode: String,
}

impl PolarQuantizer {
    pub fn new(bit_width: u8) -> Self {
        Self {
            bit_width,
            scale_mode: "per_token".to_string(),
        }
    }

    pub fn new_with_scale_mode(bit_width: u8, scale_mode: String) -> Self {
        Self {
            bit_width,
            scale_mode,
        }
    }

    pub fn quantize(&self, tensor: &Tensor) -> Result<(Tensor, Tensor)> {
        let max_val = match self.scale_mode.as_str() {
            "per_token" => {
                // Current behavior: scale across last dimension (per-token)
                tensor
                    .abs()
                    .map_err(anyhow::Error::from)?
                    .max_keepdim(tensor.rank() - 1)
                    .map_err(anyhow::Error::from)?
            }
            "per_channel" => {
                // New behavior: scale across channel dimension (typically second-to-last)
                let channel_dim = if tensor.rank() >= 2 {
                    tensor.rank() - 2
                } else {
                    0
                };
                tensor
                    .abs()
                    .map_err(anyhow::Error::from)?
                    .max_keepdim(channel_dim)
                    .map_err(anyhow::Error::from)?
            }
            _ => {
                // Fallback to per_token for unknown modes
                tensor
                    .abs()
                    .map_err(anyhow::Error::from)?
                    .max_keepdim(tensor.rank() - 1)
                    .map_err(anyhow::Error::from)?
            }
        };
        let eps = Tensor::new(&[1e-5f32], tensor.device()).map_err(anyhow::Error::from)?;
        let scale = max_val
            .broadcast_maximum(&eps)
            .map_err(anyhow::Error::from)?;

        let normalized = tensor.broadcast_div(&scale).map_err(anyhow::Error::from)?;
        let max_q = ((1 << self.bit_width) - 1) as f64;

        let shifted = normalized.affine(0.5, 0.5).map_err(anyhow::Error::from)?;
        let scaled_to_q = shifted.affine(max_q, 0.0).map_err(anyhow::Error::from)?;
        let quantized = scaled_to_q.round().map_err(anyhow::Error::from)?;

        Ok((quantized, scale))
    }

    pub fn dequantize(&self, quantized: &Tensor, scale: &Tensor) -> Result<Tensor> {
        let max_q = ((1 << self.bit_width) - 1) as f64;

        let scaled_back = quantized
            .affine(1.0 / max_q, 0.0)
            .map_err(anyhow::Error::from)?;
        let shifted_back = scaled_back.affine(2.0, -1.0).map_err(anyhow::Error::from)?;

        Ok(shifted_back
            .broadcast_mul(scale)
            .map_err(anyhow::Error::from)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_polar_quantize_dequantize() -> Result<()> {
        let device = Device::Cpu;
        let q = PolarQuantizer::new(8);

        let data = vec![-0.5f32, 0.1, 0.8, -1.2, 2.5, -0.01];
        let tensor =
            Tensor::from_vec(data.clone(), (2, 3), &device).map_err(anyhow::Error::from)?;

        let (quantized, scale) = q.quantize(&tensor)?;
        let dequantized = q.dequantize(&quantized, &scale)?;

        let deq_vec = dequantized
            .flatten_all()
            .map_err(anyhow::Error::from)?
            .to_vec1::<f32>()
            .map_err(anyhow::Error::from)?;

        for (i, (&orig, &deq)) in data.iter().zip(deq_vec.iter()).enumerate() {
            let diff = (orig - deq).abs();
            assert!(diff < 0.05, "Mismatch at {}: orig {}, deq {}", i, orig, deq);
        }

        Ok(())
    }

    #[test]
    fn test_per_channel_scaling() -> Result<()> {
        let device = Device::Cpu;
        let q_per_token = PolarQuantizer::new_with_scale_mode(8, "per_token".to_string());
        let q_per_channel = PolarQuantizer::new_with_scale_mode(8, "per_channel".to_string());

        // Create a 2x3 tensor where each row has different scales
        let data = vec![1.0f32, 2.0, 3.0, 10.0, 20.0, 30.0];
        let tensor =
            Tensor::from_vec(data.clone(), (2, 3), &device).map_err(anyhow::Error::from)?;

        // Test per_token quantization (scales across columns)
        let (quantized_pt, scale_pt) = q_per_token.quantize(&tensor)?;
        let dequantized_pt = q_per_token.dequantize(&quantized_pt, &scale_pt)?;

        // Test per_channel quantization (scales across rows)
        let (quantized_pc, scale_pc) = q_per_channel.quantize(&tensor)?;
        let dequantized_pc = q_per_channel.dequantize(&quantized_pc, &scale_pc)?;

        // Both should reconstruct reasonably well
        let deq_pt_vec = dequantized_pt
            .flatten_all()
            .map_err(anyhow::Error::from)?
            .to_vec1::<f32>()
            .map_err(anyhow::Error::from)?;
        let deq_pc_vec = dequantized_pc
            .flatten_all()
            .map_err(anyhow::Error::from)?
            .to_vec1::<f32>()
            .map_err(anyhow::Error::from)?;

        for (i, &orig) in data.iter().enumerate() {
            let diff_pt = (orig - deq_pt_vec[i]).abs();
            let diff_pc = (orig - deq_pc_vec[i]).abs();
            assert!(
                diff_pt < orig * 0.1,
                "Per-token mismatch at {}: orig {}, deq {}",
                i,
                orig,
                deq_pt_vec[i]
            );
            assert!(
                diff_pc < orig * 0.1,
                "Per-channel mismatch at {}: orig {}, deq {}",
                i,
                orig,
                deq_pc_vec[i]
            );
        }

        // Scales should have different shapes
        // Per-token: one scale per token (last dim), so shape should be (2, 1)
        // Per-channel: one scale per channel (first dim), so shape should be (1, 3)
        assert_eq!(scale_pt.dims(), &[2, 1]);
        assert_eq!(scale_pc.dims(), &[1, 3]);

        Ok(())
    }
}
