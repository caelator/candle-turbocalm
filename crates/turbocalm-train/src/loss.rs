use anyhow::{bail, Context, Result};
use candle_core::{DType, Tensor};

pub const DEFAULT_TEMPERATURE: f64 = 0.07;
const SELF_MASK_VALUE: f32 = -1e9;
const NORMALIZATION_EPSILON: f32 = 1e-6;

pub fn nt_xent_loss(anchor: &Tensor, positive: &Tensor, temperature: f64) -> Result<Tensor> {
    if anchor.rank() != 2 || positive.rank() != 2 {
        bail!("nt_xent_loss expects rank-2 anchor and positive tensors")
    }

    let (batch_size, dim) = anchor
        .dims2()
        .context("failed to read anchor embedding dimensions")?;
    let (positive_batch_size, positive_dim) = positive
        .dims2()
        .context("failed to read positive embedding dimensions")?;

    if batch_size == 0 {
        bail!("nt_xent_loss requires a non-empty batch")
    }
    if batch_size != positive_batch_size {
        bail!(
            "nt_xent_loss batch mismatch: anchor batch {batch_size}, positive batch {positive_batch_size}"
        )
    }
    if dim != positive_dim {
        bail!("nt_xent_loss feature mismatch: anchor dim {dim}, positive dim {positive_dim}")
    }
    if temperature <= 0.0 {
        bail!("temperature must be positive")
    }

    let device = anchor.device();
    let joined = Tensor::cat(&[anchor, positive], 0).context("failed to concatenate pairs")?;
    let normalized = l2_normalize_rows(&joined)?;
    let similarity = normalized
        .matmul(&normalized.transpose(0, 1)?)
        .context("failed to compute cosine similarity matrix")?;

    let total = batch_size * 2;
    let logits = {
        let temperature = Tensor::new(&[temperature as f32], device)
            .context("failed to create temperature tensor")?
            .reshape((1, 1))
            .context("failed to reshape temperature tensor")?;
        similarity
            .broadcast_div(&temperature.broadcast_as((total, total))?)
            .context("failed to apply temperature scaling")?
    };

    let diagonal_mask = {
        let row_ids = Tensor::arange(0u32, total as u32, device)
            .context("failed to create row ids")?
            .reshape((total, 1))
            .context("failed to reshape row ids")?;
        let col_ids = Tensor::arange(0u32, total as u32, device)
            .context("failed to create column ids")?
            .reshape((1, total))
            .context("failed to reshape column ids")?;
        row_ids
            .broadcast_eq(&col_ids)
            .context("failed to create diagonal mask")?
    };

    let masked_logits = diagonal_mask
        .where_cond(
            &Tensor::full(SELF_MASK_VALUE, (total, total), device)
                .context("failed to build self-similarity mask tensor")?,
            &logits,
        )
        .context("failed to mask self-similarity logits")?;

    let targets = Tensor::from_vec(
        positive_pair_indices(batch_size),
        total,
        device,
    )
    .context("failed to create NT-Xent targets")?
    .to_dtype(DType::U32)
    .context("failed to cast NT-Xent targets")?;

    candle_nn::loss::cross_entropy(&masked_logits, &targets)
        .context("failed to compute NT-Xent cross entropy")
}

fn l2_normalize_rows(embeddings: &Tensor) -> Result<Tensor> {
    let norms = embeddings
        .sqr()
        .context("failed to square embeddings for normalization")?
        .sum_keepdim(1)
        .context("failed to sum embedding norms")?
        .sqrt()
        .context("failed to sqrt embedding norms")?;

    let epsilon = Tensor::new(&[NORMALIZATION_EPSILON], embeddings.device())
        .context("failed to create normalization epsilon tensor")?
        .reshape((1, 1))
        .context("failed to reshape normalization epsilon tensor")?;
    let stabilized_norms = norms
        .broadcast_add(&epsilon.broadcast_as(norms.shape().clone())?)
        .context("failed to stabilize embedding norms")?;

    embeddings
        .broadcast_div(&stabilized_norms)
        .context("failed to normalize embeddings")
}

fn positive_pair_indices(batch_size: usize) -> Vec<u32> {
    let total = batch_size * 2;
    (0..total)
        .map(|idx| {
            if idx < batch_size {
                (idx + batch_size) as u32
            } else {
                (idx - batch_size) as u32
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Var};

    #[test]
    fn nt_xent_supports_backward() -> Result<()> {
        let device = Device::Cpu;
        let anchor = Var::from_vec(vec![1f32, 0., 0., 1.], (2, 2), &device)?;
        let positive = Var::from_vec(vec![0.9f32, 0.1, 0.1, 0.9], (2, 2), &device)?;

        let loss = nt_xent_loss(anchor.as_tensor(), positive.as_tensor(), DEFAULT_TEMPERATURE)?;
        let grads = loss.backward()?;

        assert!(grads.get(anchor.as_tensor()).is_some());
        assert!(grads.get(positive.as_tensor()).is_some());
        Ok(())
    }
}
