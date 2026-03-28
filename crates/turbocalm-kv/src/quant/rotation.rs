use candle_core::{Device, Result, Tensor};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub fn generate_orthogonal_matrix(dim: usize, seed: u64, device: &Device) -> Result<Tensor> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut basis: Vec<Vec<f32>> = Vec::with_capacity(dim);

    for _ in 0..dim {
        let mut v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        for b in &basis {
            let dot_vb: f32 = v.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let dot_bb: f32 = b.iter().map(|x| x * x).sum();
            let proj_scalar = dot_vb / dot_bb;
            for i in 0..dim {
                v[i] -= proj_scalar * b[i];
            }
        }

        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-6 {
            for i in 0..dim {
                v[i] /= norm;
            }
        } else {
            for i in 0..dim {
                v[i] = if i == basis.len() { 1.0 } else { 0.0 };
            }
        }
        basis.push(v);
    }

    let flat_basis: Vec<f32> = basis.into_iter().flatten().collect();
    Tensor::from_vec(flat_basis, (dim, dim), device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orthogonality() -> Result<()> {
        let device = Device::Cpu;
        let dim = 16;
        let q = generate_orthogonal_matrix(dim, 42, &device)?;

        let qt = q.t()?;
        let identity_approx = qt.matmul(&q)?;
        let identity_vec = identity_approx.flatten_all()?.to_vec1::<f32>()?;

        for i in 0..dim {
            for j in 0..dim {
                let val = identity_vec[i * dim + j];
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (val - expected).abs() < 1e-4,
                    "Failed at ({}, {}): val {}, expected {}",
                    i,
                    j,
                    val,
                    expected
                );
            }
        }

        Ok(())
    }
}
