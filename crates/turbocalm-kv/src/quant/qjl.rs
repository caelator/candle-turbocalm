use anyhow::Result;
use candle_core::{Device, Tensor};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub struct QjlProjector {
    pub dim: usize,
    pub projection_matrix: Tensor,
    pub threshold: f32,
}

impl QjlProjector {
    pub fn new(
        dim: usize,
        original_dim: usize,
        seed: u64,
        threshold: f32,
        device: &Device,
    ) -> Result<Self> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut data = Vec::with_capacity(dim * original_dim);

        for _ in 0..(dim * original_dim) {
            data.push(rng.gen_range(-1.0f32..1.0f32));
        }

        let projection_matrix =
            Tensor::from_vec(data, (original_dim, dim), device).map_err(anyhow::Error::from)?;

        Ok(Self {
            dim,
            projection_matrix,
            threshold,
        })
    }

    pub fn project(&self, residual: &Tensor) -> Result<(Tensor, Tensor)> {
        let projected = residual
            .matmul(&self.projection_matrix)
            .map_err(anyhow::Error::from)?;
        let threshold_tensor = Tensor::new(self.threshold, residual.device())
            .map_err(anyhow::Error::from)?
            .broadcast_as(projected.shape())
            .map_err(anyhow::Error::from)?;
        let signs = projected
            .ge(&threshold_tensor)
            .map_err(anyhow::Error::from)?
            .to_dtype(candle_core::DType::U8)
            .map_err(anyhow::Error::from)?;
        let scale = projected
            .abs()
            .map_err(anyhow::Error::from)?
            .mean_keepdim(projected.rank() - 1)
            .map_err(anyhow::Error::from)?;

        Ok((signs, scale))
    }

    pub fn reconstruct(&self, signs: &Tensor, scale: &Tensor) -> Result<Tensor> {
        let f_signs = signs
            .to_dtype(candle_core::DType::F32)
            .map_err(anyhow::Error::from)?;
        let mapped_signs = f_signs.affine(2.0, -1.0).map_err(anyhow::Error::from)?;
        let scaled_projection = mapped_signs
            .broadcast_mul(scale)
            .map_err(anyhow::Error::from)?;

        Ok(scaled_projection
            .matmul(&self.projection_matrix.t().map_err(anyhow::Error::from)?)
            .map_err(anyhow::Error::from)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qjl_sign_preservation() -> Result<()> {
        let device = Device::Cpu;
        let projector = QjlProjector::new(16, 32, 123, 0.0, &device)?;

        let data: Vec<f32> = (0..64).map(|x| (x as f32 - 32.0) / 10.0).collect();
        let residual = Tensor::from_vec(data, (2, 32), &device).map_err(anyhow::Error::from)?;

        let (signs, scale) = projector.project(&residual)?;

        let signs_vec = signs
            .flatten_all()
            .map_err(anyhow::Error::from)?
            .to_vec1::<u8>()
            .map_err(anyhow::Error::from)?;
        for &s in &signs_vec {
            assert!(s == 0 || s == 1);
        }

        let reconstructed = projector.reconstruct(&signs, &scale)?;
        assert_eq!(reconstructed.dims(), &[2, 32]);

        Ok(())
    }
}
