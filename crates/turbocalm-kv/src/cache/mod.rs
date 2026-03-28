pub mod dense;

use candle_core::{DType, Device, Result, Tensor};
use crate::quant::pack::{pack_bits, unpack_bits};
use crate::quant::profile::QuantProfile;
use crate::quant::polar::PolarQuantizer;
use crate::quant::qjl::QjlProjector;
use crate::quant::rotation::generate_orthogonal_matrix;

pub trait KvCache {
    fn append(&mut self, key: &Tensor, value: &Tensor) -> Result<()>;
    fn get_key(&mut self) -> Result<Tensor>;
    fn get_value(&mut self) -> Result<Tensor>;
    fn clear(&mut self);
}

#[derive(Default)]
struct IncrementalTensorCache {
    concatenated: Option<Tensor>,
    decompressed_entries: usize,
}

#[derive(Default)]
struct UncompressedCache {
    key: IncrementalTensorCache,
    value: IncrementalTensorCache,
}

pub struct CompressedTensor {
    quantized: Tensor,
    quantized_shape: Vec<usize>,
    q_scale: Tensor,
    signs: Tensor,
    r_scale: Tensor,
    original_shape: Vec<usize>,
}

pub struct TurboKvCache {
    profile: QuantProfile,
    polar_quantizer: PolarQuantizer,
    keys: Vec<CompressedTensor>,
    values: Vec<CompressedTensor>,
    rotation_matrix: Option<Tensor>,
    qjl_projector: Option<QjlProjector>,
    uncompressed_cache: UncompressedCache,
}

impl TurboKvCache {
    pub fn new(profile: QuantProfile) -> Self {
        let polar_quantizer = PolarQuantizer::new(profile.bit_width);
        Self { 
            profile,
            polar_quantizer,
            keys: Vec::new(),
            values: Vec::new(),
            rotation_matrix: None,
            qjl_projector: None,
            uncompressed_cache: UncompressedCache::default(),
        }
    }

    fn init_if_needed(&mut self, dim: usize, device: &Device) -> Result<()> {
        if self.rotation_matrix.is_none() {
            let rot = generate_orthogonal_matrix(dim, self.profile.rotation_seed, device)?;
            self.rotation_matrix = Some(rot);
        }
        if self.qjl_projector.is_none() {
            let proj = QjlProjector::new(
                self.profile.qjl_dim,
                dim,
                self.profile.rotation_seed,
                self.profile.qjl_threshold,
                device
            )?;
            self.qjl_projector = Some(proj);
        }
        Ok(())
    }

    fn flatten_for_matmul(tensor: &Tensor) -> Result<(Tensor, Vec<usize>)> {
        let shape = tensor.dims().to_vec();
        let rank = shape.len();
        if rank <= 2 {
            return Ok((tensor.clone(), shape));
        }
        let dim = shape[rank - 1];
        let num_elements: usize = shape[0..rank - 1].iter().product();
        let reshaped = tensor.reshape((num_elements, dim))?;
        Ok((reshaped, shape))
    }

    fn unflatten_after_matmul(tensor: &Tensor, original_shape: &[usize], new_last_dim: usize) -> Result<Tensor> {
        let mut new_shape = original_shape.to_vec();
        let len = new_shape.len();
        new_shape[len - 1] = new_last_dim;
        tensor.reshape(new_shape.as_slice())
    }

    fn compress_tensor(&mut self, tensor: &Tensor) -> Result<CompressedTensor> {
        let dims = tensor.dims();
        let dim = *dims.last().ok_or_else(|| candle_core::Error::Msg("Empty tensor".to_string()))?;
        self.init_if_needed(dim, tensor.device())?;

        let rot = self.rotation_matrix.as_ref().unwrap();
        let qjl = self.qjl_projector.as_ref().unwrap();

        // Flatten tensor to 2D for matmul
        let (flat_tensor, original_shape) = Self::flatten_for_matmul(tensor)?;

        // Apply rotation
        let rotated = flat_tensor.matmul(rot)?;

        // Polar quantization
        let (quantized_flat, q_scale_flat) = self.polar_quantizer.quantize(&rotated)?;
        let quantized_shape = quantized_flat.dims().to_vec();
        let packed_quantized = pack_bits(
            &quantized_flat.to_dtype(DType::U8)?,
            self.profile.bit_width,
        )?;

        // Compute residual
        let dequantized_flat = self.polar_quantizer.dequantize(&quantized_flat, &q_scale_flat)?;
        let residual_flat = rotated.broadcast_sub(&dequantized_flat)?;

        // QJL projection
        let (signs_flat, r_scale_flat) = qjl.project(&residual_flat)?;

        Ok(CompressedTensor {
            quantized: packed_quantized,
            quantized_shape,
            q_scale: q_scale_flat,
            signs: signs_flat,
            r_scale: r_scale_flat,
            original_shape,
        })
    }

    fn decompress_tensor_with(
        polar_quantizer: &PolarQuantizer,
        bit_width: u8,
        rot: &Tensor,
        qjl: &QjlProjector,
        comp: &CompressedTensor,
    ) -> Result<Tensor> {
        // Reconstruct from polar
        let unpacked_quantized = unpack_bits(&comp.quantized, bit_width, &comp.quantized_shape)?
            .to_dtype(comp.q_scale.dtype())?;
        let dequantized_flat = polar_quantizer.dequantize(&unpacked_quantized, &comp.q_scale)?;

        // Reconstruct from QJL
        let recon_residual_flat = qjl.reconstruct(&comp.signs, &comp.r_scale)?;

        // Add residual
        let reconstructed_rotated_flat = dequantized_flat.broadcast_add(&recon_residual_flat)?;

        // Inverse rotation (rot is orthogonal, so rot.t() is inverse)
        let rot_t = rot.t()?;
        let reconstructed_flat = reconstructed_rotated_flat.matmul(&rot_t)?;

        // Unflatten
        let dim = comp.original_shape.last().unwrap();
        Self::unflatten_after_matmul(&reconstructed_flat, &comp.original_shape, *dim)
    }

    fn decompress_tensor(&self, comp: &CompressedTensor) -> Result<Tensor> {
        let rot = self.rotation_matrix.as_ref().ok_or_else(|| candle_core::Error::Msg("Uninitialized".to_string()))?;
        let qjl = self.qjl_projector.as_ref().ok_or_else(|| candle_core::Error::Msg("Uninitialized".to_string()))?;
        Self::decompress_tensor_with(
            &self.polar_quantizer,
            self.profile.bit_width,
            rot,
            qjl,
            comp,
        )
    }

    fn cat_dim(rank: usize) -> Result<usize> {
        if rank < 1 {
            return Err(candle_core::Error::Msg("Invalid tensor rank".to_string()));
        }
        Ok(if rank >= 2 { rank - 2 } else { 0 })
    }

    fn concat_tensors(tensors: &[Tensor]) -> Result<Tensor> {
        if tensors.is_empty() {
            return Err(candle_core::Error::Msg("KV Cache is empty".to_string()));
        }
        if tensors.len() == 1 {
            return Ok(tensors[0].clone());
        }
        let cat_dim = Self::cat_dim(tensors[0].dims().len())?;
        let refs = tensors.iter().collect::<Vec<_>>();
        Tensor::cat(&refs, cat_dim)
    }

    fn update_uncompressed_cache(
        polar_quantizer: &PolarQuantizer,
        bit_width: u8,
        rot: &Tensor,
        qjl: &QjlProjector,
        compressed: &[CompressedTensor],
        cache: &mut IncrementalTensorCache,
    ) -> Result<Tensor> {
        if compressed.is_empty() {
            return Err(candle_core::Error::Msg("KV Cache is empty".to_string()));
        }

        if cache.decompressed_entries < compressed.len() {
            let mut new_tensors = Vec::with_capacity(compressed.len() - cache.decompressed_entries);
            for comp in &compressed[cache.decompressed_entries..] {
                new_tensors.push(Self::decompress_tensor_with(
                    polar_quantizer,
                    bit_width,
                    rot,
                    qjl,
                    comp,
                )?);
            }

            let new_suffix = Self::concat_tensors(&new_tensors)?;
            cache.concatenated = Some(match &cache.concatenated {
                Some(prefix) => {
                    let cat_dim = Self::cat_dim(prefix.dims().len())?;
                    Tensor::cat(&[prefix, &new_suffix], cat_dim)?
                }
                None => new_suffix,
            });
            cache.decompressed_entries = compressed.len();
        }

        cache
            .concatenated
            .as_ref()
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("KV Cache is empty".to_string()))
    }
}

impl KvCache for TurboKvCache {
    fn append(&mut self, key: &Tensor, value: &Tensor) -> Result<()> {
        let compressed_key = self.compress_tensor(key)?;
        let compressed_value = self.compress_tensor(value)?;
        self.keys.push(compressed_key);
        self.values.push(compressed_value);
        Ok(())
    }

    fn get_key(&mut self) -> Result<Tensor> {
        let rot = self.rotation_matrix.as_ref().ok_or_else(|| candle_core::Error::Msg("Uninitialized".to_string()))?;
        let qjl = self.qjl_projector.as_ref().ok_or_else(|| candle_core::Error::Msg("Uninitialized".to_string()))?;
        Self::update_uncompressed_cache(
            &self.polar_quantizer,
            self.profile.bit_width,
            rot,
            qjl,
            &self.keys,
            &mut self.uncompressed_cache.key,
        )
    }

    fn get_value(&mut self) -> Result<Tensor> {
        let rot = self.rotation_matrix.as_ref().ok_or_else(|| candle_core::Error::Msg("Uninitialized".to_string()))?;
        let qjl = self.qjl_projector.as_ref().ok_or_else(|| candle_core::Error::Msg("Uninitialized".to_string()))?;
        Self::update_uncompressed_cache(
            &self.polar_quantizer,
            self.profile.bit_width,
            rot,
            qjl,
            &self.values,
            &mut self.uncompressed_cache.value,
        )
    }

    fn clear(&mut self) {
        self.keys.clear();
        self.values.clear();
        self.uncompressed_cache = UncompressedCache::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use crate::cache::dense::DenseKvCache;

    fn generate_random_tensor(shape: &[usize], device: &Device) -> Result<Tensor> {
        let num_elements: usize = shape.iter().product();
        let mut data = Vec::with_capacity(num_elements);
        for i in 0..num_elements {
            data.push((i as f32 * 0.123).sin() * 2.0);
        }
        Tensor::from_vec(data, shape, device)
    }

    #[test]
    fn test_turbo_kv_cache_roundtrip_and_memory() -> Result<()> {
        let device = Device::Cpu;
        let profile = QuantProfile {
            bit_width: 8,
            rotation_seed: 42,
            qjl_dim: 8,
            qjl_threshold: 0.0,
            scale_mode: "per_token".to_string(),
        };

        let mut turbo_cache = TurboKvCache::new(profile.clone());
        let mut dense_cache = DenseKvCache::new();

        let shape = vec![2, 4, 16, 32];
        let key = generate_random_tensor(&shape, &device)?;
        let value = generate_random_tensor(&shape, &device)?;

        turbo_cache.append(&key, &value)?;
        dense_cache.append(&key, &value)?;

        let retrieved_key = turbo_cache.get_key()?;
        let expected_key = dense_cache.get_key()?;

        // MSE check
        let diff = retrieved_key.broadcast_sub(&expected_key)?;
        let diff_sq = diff.broadcast_mul(&diff)?;
        let mse = diff_sq.mean_all()?.to_scalar::<f32>()?;
        assert!(mse < 0.25, "MSE is too high: {}", mse); // Adjust threshold based on compression loss. 8-bit usually has very small loss. QJL adds some error too.

        // Memory usage check (elements count approximation)
        let dense_bytes = key.elem_count() * 4;
        
        let c_key = &turbo_cache.keys[0];
        let turbo_conceptual_bytes = 
            (c_key.quantized.elem_count() * profile.bit_width as usize / 8) + 
            (c_key.q_scale.elem_count() * 4) +
            (c_key.signs.elem_count() * 1) + 
            (c_key.r_scale.elem_count() * 4);

        assert!(turbo_conceptual_bytes < dense_bytes, "TurboCache conceptual bytes ({}) should be less than dense bytes ({})", turbo_conceptual_bytes, dense_bytes);

        Ok(())
    }
}
