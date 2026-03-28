use anyhow::{Context, Result};
use candle_core::{DType, Device, Shape, Tensor};
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

/// PyTorch state dict parser that can handle both .bin and safetensors files
pub struct StateDictParser {
    device: Device,
}

impl StateDictParser {
    /// Create a new state dict parser
    pub fn new(device: Device) -> Self {
        Self { device }
    }

    /// Parse a model from multiple checkpoint files
    pub fn parse_model_files(&self, file_paths: &[PathBuf]) -> Result<HashMap<String, Tensor>> {
        info!("Parsing {} model files", file_paths.len());

        let mut all_tensors = HashMap::new();

        for (i, path) in file_paths.iter().enumerate() {
            debug!("Processing file {}/{}: {}", i + 1, file_paths.len(), path.display());

            let file_tensors = self.parse_single_file(path)
                .with_context(|| format!("Failed to parse file: {}", path.display()))?;

            info!("Loaded {} tensors from {}", file_tensors.len(), path.display());

            // Check for tensor name conflicts
            for (name, tensor) in file_tensors {
                if all_tensors.contains_key(&name) {
                    warn!("Duplicate tensor name found: {} (overwriting)", name);
                }
                all_tensors.insert(name, tensor);
            }
        }

        info!("Total tensors loaded: {}", all_tensors.len());
        Ok(all_tensors)
    }

    /// Parse a single checkpoint file (auto-detect format)
    pub fn parse_single_file<P: AsRef<Path>>(&self, path: P) -> Result<HashMap<String, Tensor>> {
        let path = path.as_ref();
        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("");

        match extension {
            "safetensors" => self.parse_safetensors(path),
            "bin" => self.parse_pytorch_bin(path),
            _ => {
                // Try to auto-detect based on file content
                if self.is_safetensors_file(path)? {
                    self.parse_safetensors(path)
                } else {
                    self.parse_pytorch_bin(path)
                }
            }
        }
    }

    /// Parse a safetensors file
    pub fn parse_safetensors<P: AsRef<Path>>(&self, path: P) -> Result<HashMap<String, Tensor>> {
        let path = path.as_ref();
        debug!("Parsing safetensors file: {}", path.display());

        let content = std::fs::read(path)
            .with_context(|| format!("Failed to read safetensors file: {}", path.display()))?;

        let safetensors = SafeTensors::deserialize(&content)
            .with_context(|| format!("Failed to deserialize safetensors: {}", path.display()))?;

        let mut tensors = HashMap::new();

        for tensor_name in safetensors.names() {
            let tensor_view = safetensors
                .tensor(tensor_name)
                .with_context(|| format!("Failed to get tensor: {}", tensor_name))?;

            let tensor = self.safetensor_view_to_candle_tensor(tensor_view, tensor_name)?;
            tensors.insert(tensor_name.to_string(), tensor);
        }

        debug!("Successfully parsed {} tensors from safetensors", tensors.len());
        Ok(tensors)
    }

    /// Parse a PyTorch .bin file (this is a simplified implementation)
    pub fn parse_pytorch_bin<P: AsRef<Path>>(&self, path: P) -> Result<HashMap<String, Tensor>> {
        let path = path.as_ref();
        debug!("Parsing PyTorch .bin file: {}", path.display());

        // For PyTorch .bin files, we'll use candle's built-in PyTorch loading
        // This is a simplified approach - in practice you might need more sophisticated handling

        // Try to load as if it were a safetensors file first (some .bin files are actually safetensors)
        if let Ok(tensors) = self.parse_safetensors(path) {
            info!("PyTorch .bin file was actually safetensors format");
            return Ok(tensors);
        }

        // If not safetensors, we'd need to implement proper PyTorch loading
        // For now, return an error with suggestion
        Err(anyhow::anyhow!(
            "PyTorch .bin loading not fully implemented. Consider converting to safetensors first. File: {}",
            path.display()
        ))
    }

    /// Check if a file is in safetensors format
    fn is_safetensors_file<P: AsRef<Path>>(&self, path: P) -> Result<bool> {
        let content = std::fs::read(path)?;

        // Simple heuristic: try to deserialize as safetensors
        match SafeTensors::deserialize(&content) {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    /// Convert a safetensors tensor view to a candle tensor
    fn safetensor_view_to_candle_tensor(
        &self,
        tensor_view: safetensors::tensor::TensorView,
        tensor_name: &str,
    ) -> Result<Tensor> {
        // Convert safetensors dtype to candle dtype
        let candle_dtype = self.convert_dtype(tensor_view.dtype())
            .with_context(|| format!("Unsupported dtype for tensor: {}", tensor_name))?;

        // Convert shape
        let shape = Shape::from_dims(tensor_view.shape());

        // Get the data as bytes
        let data = tensor_view.data();

        // Create tensor from raw data
        let tensor = match candle_dtype {
            DType::F32 => {
                let float_data = self.bytes_to_f32_slice(data)?;
                Tensor::from_slice(float_data, shape, &self.device)?
            }
            DType::F16 => {
                let half_data = self.bytes_to_f16_slice(data)?;
                Tensor::from_slice(half_data, shape, &self.device)?
            }
            DType::BF16 => {
                let bf16_data = self.bytes_to_bf16_slice(data)?;
                Tensor::from_slice(bf16_data, shape, &self.device)?
            }
            DType::I64 => {
                let int_data = self.bytes_to_i64_slice(data)?;
                Tensor::from_slice(int_data, shape, &self.device)?
            }
            DType::U32 => {
                let uint_data = self.bytes_to_u32_slice(data)?;
                Tensor::from_slice(uint_data, shape, &self.device)?
            }
            DType::U8 => {
                Tensor::from_slice(data, shape, &self.device)?
            }
            DType::F64 => {
                let double_data = self.bytes_to_f64_slice(data)?;
                Tensor::from_slice(double_data, shape, &self.device)?
            }
        };

        debug!("Loaded tensor '{}': {:?}", tensor_name, tensor.shape());
        Ok(tensor)
    }

    /// Convert safetensors dtype to candle dtype
    fn convert_dtype(&self, safetensors_dtype: safetensors::Dtype) -> Result<DType> {
        match safetensors_dtype {
            safetensors::Dtype::F32 => Ok(DType::F32),
            safetensors::Dtype::F16 => Ok(DType::F16),
            safetensors::Dtype::BF16 => Ok(DType::BF16),
            safetensors::Dtype::I64 => Ok(DType::I64),
            safetensors::Dtype::U32 => Ok(DType::U32),
            safetensors::Dtype::U8 => Ok(DType::U8),
            safetensors::Dtype::F64 => Ok(DType::F64),
            _ => Err(anyhow::anyhow!("Unsupported dtype: {:?}", safetensors_dtype)),
        }
    }

    /// Convert bytes to f32 slice
    fn bytes_to_f32_slice(&self, bytes: &[u8]) -> Result<&[f32]> {
        if bytes.len() % 4 != 0 {
            return Err(anyhow::anyhow!("Invalid byte length for f32 data"));
        }
        Ok(unsafe {
            std::slice::from_raw_parts(
                bytes.as_ptr() as *const f32,
                bytes.len() / 4,
            )
        })
    }

    /// Convert bytes to f16 slice
    fn bytes_to_f16_slice(&self, bytes: &[u8]) -> Result<&[half::f16]> {
        if bytes.len() % 2 != 0 {
            return Err(anyhow::anyhow!("Invalid byte length for f16 data"));
        }
        Ok(unsafe {
            std::slice::from_raw_parts(
                bytes.as_ptr() as *const half::f16,
                bytes.len() / 2,
            )
        })
    }

    /// Convert bytes to bf16 slice
    fn bytes_to_bf16_slice(&self, bytes: &[u8]) -> Result<&[half::bf16]> {
        if bytes.len() % 2 != 0 {
            return Err(anyhow::anyhow!("Invalid byte length for bf16 data"));
        }
        Ok(unsafe {
            std::slice::from_raw_parts(
                bytes.as_ptr() as *const half::bf16,
                bytes.len() / 2,
            )
        })
    }

    /// Convert bytes to i64 slice
    fn bytes_to_i64_slice(&self, bytes: &[u8]) -> Result<&[i64]> {
        if bytes.len() % 8 != 0 {
            return Err(anyhow::anyhow!("Invalid byte length for i64 data"));
        }
        Ok(unsafe {
            std::slice::from_raw_parts(
                bytes.as_ptr() as *const i64,
                bytes.len() / 8,
            )
        })
    }

    /// Convert bytes to u32 slice
    fn bytes_to_u32_slice(&self, bytes: &[u8]) -> Result<&[u32]> {
        if bytes.len() % 4 != 0 {
            return Err(anyhow::anyhow!("Invalid byte length for u32 data"));
        }
        Ok(unsafe {
            std::slice::from_raw_parts(
                bytes.as_ptr() as *const u32,
                bytes.len() / 4,
            )
        })
    }

    /// Convert bytes to f64 slice
    fn bytes_to_f64_slice(&self, bytes: &[u8]) -> Result<&[f64]> {
        if bytes.len() % 8 != 0 {
            return Err(anyhow::anyhow!("Invalid byte length for f64 data"));
        }
        Ok(unsafe {
            std::slice::from_raw_parts(
                bytes.as_ptr() as *const f64,
                bytes.len() / 8,
            )
        })
    }

    /// Get summary information about parsed tensors
    pub fn get_tensor_summary(&self, tensors: &HashMap<String, Tensor>) -> TensorSummary {
        let mut total_parameters = 0usize;
        let mut total_size_bytes = 0usize;
        let mut dtype_counts: HashMap<String, usize> = HashMap::new();

        for (name, tensor) in tensors {
            let param_count = tensor.elem_count();
            let dtype_size = match tensor.dtype() {
                DType::F32 => 4,
                DType::F16 => 2,
                DType::BF16 => 2,
                DType::F64 => 8,
                DType::I64 => 8,
                DType::U32 => 4,
                DType::U8 => 1,
            };

            total_parameters += param_count;
            total_size_bytes += param_count * dtype_size;

            let dtype_str = format!("{:?}", tensor.dtype());
            *dtype_counts.entry(dtype_str).or_insert(0) += 1;
        }

        TensorSummary {
            tensor_count: tensors.len(),
            total_parameters,
            total_size_mb: total_size_bytes as f64 / (1024.0 * 1024.0),
            dtype_distribution: dtype_counts,
        }
    }
}

/// Summary information about a collection of tensors
#[derive(Debug, Clone)]
pub struct TensorSummary {
    pub tensor_count: usize,
    pub total_parameters: usize,
    pub total_size_mb: f64,
    pub dtype_distribution: HashMap<String, usize>,
}

impl TensorSummary {
    /// Display a formatted summary
    pub fn display_summary(&self) {
        info!("Tensor Summary:");
        info!("  Total tensors: {}", self.tensor_count);
        info!("  Total parameters: {:.2}M", self.total_parameters as f64 / 1_000_000.0);
        info!("  Total size: {:.1} MB", self.total_size_mb);
        info!("  Data type distribution:");
        for (dtype, count) in &self.dtype_distribution {
            info!("    {}: {} tensors", dtype, count);
        }
    }
}

/// Convenience functions for common parsing operations
pub mod convenience {
    use super::*;

    /// Quick function to parse model files with default device
    pub fn parse_model_files(file_paths: &[PathBuf]) -> Result<HashMap<String, Tensor>> {
        let device = Device::Cpu; // Default to CPU
        let parser = StateDictParser::new(device);
        parser.parse_model_files(file_paths)
    }

    /// Quick function to parse a single file
    pub fn parse_single_file<P: AsRef<Path>>(path: P) -> Result<HashMap<String, Tensor>> {
        let device = Device::Cpu;
        let parser = StateDictParser::new(device);
        parser.parse_single_file(path)
    }

    /// Parse and summarize a model
    pub fn parse_and_summarize(file_paths: &[PathBuf]) -> Result<(HashMap<String, Tensor>, TensorSummary)> {
        let parser = StateDictParser::new(Device::Cpu);
        let tensors = parser.parse_model_files(file_paths)?;
        let summary = parser.get_tensor_summary(&tensors);
        Ok((tensors, summary))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_dict_parser_creation() {
        let parser = StateDictParser::new(Device::Cpu);
        // Test basic functionality without requiring actual files
        assert!(parser.device.is_cpu());
    }

    #[test]
    fn test_dtype_conversion() {
        let parser = StateDictParser::new(Device::Cpu);

        assert_eq!(parser.convert_dtype(safetensors::Dtype::F32).unwrap(), DType::F32);
        assert_eq!(parser.convert_dtype(safetensors::Dtype::F16).unwrap(), DType::F16);
        assert_eq!(parser.convert_dtype(safetensors::Dtype::BF16).unwrap(), DType::BF16);
        assert_eq!(parser.convert_dtype(safetensors::Dtype::I64).unwrap(), DType::I64);
        assert_eq!(parser.convert_dtype(safetensors::Dtype::U32).unwrap(), DType::U32);
        assert_eq!(parser.convert_dtype(safetensors::Dtype::U8).unwrap(), DType::U8);
    }

    #[test]
    fn test_tensor_summary() {
        let mut tensors = HashMap::new();

        // Create a test tensor
        let test_tensor = Tensor::zeros((100, 200), DType::F32, &Device::Cpu).unwrap();
        tensors.insert("test_weight".to_string(), test_tensor);

        let parser = StateDictParser::new(Device::Cpu);
        let summary = parser.get_tensor_summary(&tensors);

        assert_eq!(summary.tensor_count, 1);
        assert_eq!(summary.total_parameters, 20000); // 100 * 200
        assert!((summary.total_size_mb - 0.076).abs() < 0.01); // 20000 * 4 bytes ≈ 0.076 MB

        summary.display_summary(); // Test that this doesn't panic
    }

    #[test]
    fn test_bytes_conversion() {
        let parser = StateDictParser::new(Device::Cpu);

        // Test f32 conversion
        let f32_bytes = [0u8, 0, 128, 63]; // 1.0 in IEEE 754 little-endian
        let f32_slice = parser.bytes_to_f32_slice(&f32_bytes).unwrap();
        assert_eq!(f32_slice.len(), 1);

        // Test error case
        let invalid_bytes = [0u8, 0, 128]; // 3 bytes, not divisible by 4
        assert!(parser.bytes_to_f32_slice(&invalid_bytes).is_err());
    }
}