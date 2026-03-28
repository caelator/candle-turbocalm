use candle_core::{Result, Tensor};

pub struct PolarQuantizer {
    pub bit_width: u8,
}

impl PolarQuantizer {
    pub fn new(bit_width: u8) -> Self {
        Self { bit_width }
    }

    pub fn quantize(&self, tensor: &Tensor) -> Result<(Tensor, Tensor)> {
        let max_val = tensor.abs()?.max_keepdim(tensor.rank() - 1)?;
        let eps = Tensor::new(&[1e-5f32], tensor.device())?;
        let scale = max_val.broadcast_maximum(&eps)?;

        let normalized = tensor.broadcast_div(&scale)?;
        let max_q = ((1 << self.bit_width) - 1) as f64;

        let shifted = normalized.affine(0.5, 0.5)?;
        let scaled_to_q = shifted.affine(max_q, 0.0)?;
        let quantized = scaled_to_q.round()?;

        Ok((quantized, scale))
    }

    pub fn dequantize(&self, quantized: &Tensor, scale: &Tensor) -> Result<Tensor> {
        let max_q = ((1 << self.bit_width) - 1) as f64;

        let scaled_back = quantized.affine(1.0 / max_q, 0.0)?;
        let shifted_back = scaled_back.affine(2.0, -1.0)?;

        shifted_back.broadcast_mul(scale)
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
        let tensor = Tensor::from_vec(data.clone(), (2, 3), &device)?;

        let (quantized, scale) = q.quantize(&tensor)?;
        let dequantized = q.dequantize(&quantized, &scale)?;

        let deq_vec = dequantized.flatten_all()?.to_vec1::<f32>()?;

        for (i, (&orig, &deq)) in data.iter().zip(deq_vec.iter()).enumerate() {
            let diff = (orig - deq).abs();
            assert!(diff < 0.05, "Mismatch at {}: orig {}, deq {}", i, orig, deq);
        }

        Ok(())
    }
}
