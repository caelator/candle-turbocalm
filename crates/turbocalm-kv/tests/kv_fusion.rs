use anyhow::Result;
use candle_core::{Device, Tensor};
use turbocalm_core::QuantProfile;
use turbocalm_kv::cache::{KvCache, TurboKvCache};

fn synthetic_tensor(seq_len: usize, offset: f32, device: &Device) -> Result<Tensor> {
    let batch = 1;
    let heads = 2;
    let head_dim = 4;
    let mut data = Vec::with_capacity(batch * heads * seq_len * head_dim);
    for batch_index in 0..batch {
        for head_index in 0..heads {
            for seq_index in 0..seq_len {
                for dim_index in 0..head_dim {
                    data.push(
                        offset
                            + batch_index as f32 * 0.05
                            + head_index as f32 * 0.5
                            + seq_index as f32 * 0.25
                            + dim_index as f32 * 0.1,
                    );
                }
            }
        }
    }
    Tensor::from_vec(data, (batch, heads, seq_len, head_dim), device).map_err(anyhow::Error::from)
}

fn mean_pool_reference(tensor: &Tensor, target_seq_len: usize) -> Result<Tensor> {
    let seq_dim = tensor.rank() - 2;
    let seq_len = tensor.dims()[seq_dim];
    anyhow::ensure!(target_seq_len > 0, "target_seq_len must be positive");
    anyhow::ensure!(
        target_seq_len <= seq_len,
        "cannot pool sequence length {} to {}",
        seq_len,
        target_seq_len
    );

    if target_seq_len == seq_len {
        return Ok(tensor.clone());
    }

    let mut pooled = Vec::with_capacity(target_seq_len);
    for bucket in 0..target_seq_len {
        let start = bucket * seq_len / target_seq_len;
        let end = (bucket + 1) * seq_len / target_seq_len;
        let chunk = tensor.narrow(seq_dim, start, end - start)?;
        pooled.push(chunk.mean(seq_dim)?);
    }

    let pooled_refs = pooled.iter().collect::<Vec<_>>();
    Tensor::stack(&pooled_refs, seq_dim).map_err(anyhow::Error::from)
}

fn mse(lhs: &Tensor, rhs: &Tensor) -> Result<f32> {
    let diff = lhs.broadcast_sub(rhs)?;
    let squared = diff.broadcast_mul(&diff)?;
    squared
        .mean_all()?
        .to_scalar::<f32>()
        .map_err(anyhow::Error::from)
}

#[test]
fn fuse_caches_mean_pools_keys_and_values() -> Result<()> {
    let device = Device::Cpu;
    let profile = QuantProfile {
        bit_width: 8,
        rotation_seed: 7,
        qjl_dim: 16,
        qjl_threshold: 0.0,
        scale_mode: "per_token".to_string(),
        clipping_percentile: 0.99,
        scale_multiplier: 1.0,
    };

    let key_a = synthetic_tensor(2, 0.25, &device)?;
    let value_a = synthetic_tensor(2, -0.5, &device)?;
    let key_b = synthetic_tensor(3, 1.5, &device)?;
    let value_b = synthetic_tensor(3, 2.25, &device)?;

    let mut cache_a = TurboKvCache::new(profile.clone());
    cache_a.append(&key_a, &value_a)?;
    let mut cache_b = TurboKvCache::new(profile);
    cache_b.append(&key_b, &value_b)?;

    let reference_key_a = cache_a.get_key()?;
    let reference_value_a = cache_a.get_value()?;
    let reference_key_b = cache_b.get_key()?;
    let reference_value_b = cache_b.get_value()?;

    let target_seq_len = 3;
    let reference_key = mean_pool_reference(
        &Tensor::cat(&[&reference_key_a, &reference_key_b], 2)?,
        target_seq_len,
    )?;
    let reference_value = mean_pool_reference(
        &Tensor::cat(&[&reference_value_a, &reference_value_b], 2)?,
        target_seq_len,
    )?;

    let mut fused = TurboKvCache::fuse_caches(&[&cache_a, &cache_b], target_seq_len)?;
    let fused_key = fused.get_key()?;
    let fused_value = fused.get_value()?;

    assert_eq!(fused_key.dims(), &[1, 2, target_seq_len, 4]);
    assert_eq!(fused_value.dims(), &[1, 2, target_seq_len, 4]);

    let key_mse = mse(&fused_key, &reference_key)?;
    let value_mse = mse(&fused_value, &reference_value)?;
    assert!(key_mse < 0.35, "fused key MSE too high: {key_mse}");
    assert!(value_mse < 0.35, "fused value MSE too high: {value_mse}");

    Ok(())
}
