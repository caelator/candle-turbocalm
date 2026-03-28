use crate::error::{Result, TensorError, TurboCALMError};
use candle_core::{Device, Tensor};
use std::collections::HashMap;
use tracing::{debug, info, warn};

/// Similarity metrics for tensor comparisons
pub struct SimilarityMetrics;

impl SimilarityMetrics {
    /// Compute cosine similarity between two tensors
    ///
    /// The tensors must have the same shape. Returns a value between -1 and 1,
    /// where 1 indicates identical tensors and -1 indicates opposite tensors.
    pub fn cosine_similarity(a: &Tensor, b: &Tensor) -> Result<f32> {
        if a.shape() != b.shape() {
            return Err(TurboCALMError::Tensor(TensorError::ShapeMismatch {
                expected: format!("{:?}", a.shape()),
                found: format!("{:?}", b.shape()),
            }));
        }

        // Flatten tensors for dot product computation
        let a_flat = a.flatten_all().map_err(|e| TurboCALMError::Candle(e))?;
        let b_flat = b.flatten_all().map_err(|e| TurboCALMError::Candle(e))?;

        // Compute dot product
        let dot_product = (&a_flat * &b_flat)
            .map_err(|e| TurboCALMError::Candle(e))?
            .sum_all()
            .map_err(|e| TurboCALMError::Candle(e))?
            .to_scalar::<f32>()
            .map_err(|e| TurboCALMError::Candle(e))?;

        // Compute norms
        let a_norm = (&a_flat * &a_flat)
            .map_err(|e| TurboCALMError::Candle(e))?
            .sum_all()
            .map_err(|e| TurboCALMError::Candle(e))?
            .sqrt()
            .map_err(|e| TurboCALMError::Candle(e))?
            .to_scalar::<f32>()
            .map_err(|e| TurboCALMError::Candle(e))?;

        let b_norm = (&b_flat * &b_flat)
            .map_err(|e| TurboCALMError::Candle(e))?
            .sum_all()
            .map_err(|e| TurboCALMError::Candle(e))?
            .sqrt()
            .map_err(|e| TurboCALMError::Candle(e))?
            .to_scalar::<f32>()
            .map_err(|e| TurboCALMError::Candle(e))?;

        // Avoid division by zero
        if a_norm == 0.0 || b_norm == 0.0 {
            warn!("One of the tensors has zero norm, cosine similarity undefined");
            return Ok(0.0);
        }

        let cosine_sim = dot_product / (a_norm * b_norm);
        Ok(cosine_sim)
    }

    /// Compute mean squared error between two tensors
    pub fn mse(a: &Tensor, b: &Tensor) -> Result<f32> {
        if a.shape() != b.shape() {
            return Err(TurboCALMError::Tensor(TensorError::ShapeMismatch {
                expected: format!("{:?}", a.shape()),
                found: format!("{:?}", b.shape()),
            }));
        }

        let diff = (a - b).map_err(|e| TurboCALMError::Candle(e))?;
        let squared_diff = (&diff * &diff).map_err(|e| TurboCALMError::Candle(e))?;
        let mse_value = squared_diff
            .mean_all()
            .map_err(|e| TurboCALMError::Candle(e))?
            .to_scalar::<f32>()
            .map_err(|e| TurboCALMError::Candle(e))?;

        Ok(mse_value)
    }

    /// Compute root mean squared error between two tensors
    pub fn rmse(a: &Tensor, b: &Tensor) -> Result<f32> {
        let mse_value = Self::mse(a, b)?;
        Ok(mse_value.sqrt())
    }

    /// Compute mean absolute error between two tensors
    pub fn mae(a: &Tensor, b: &Tensor) -> Result<f32> {
        if a.shape() != b.shape() {
            return Err(TurboCALMError::Tensor(TensorError::ShapeMismatch {
                expected: format!("{:?}", a.shape()),
                found: format!("{:?}", b.shape()),
            }));
        }

        let diff = (a - b).map_err(|e| TurboCALMError::Candle(e))?;
        let abs_diff = diff.abs().map_err(|e| TurboCALMError::Candle(e))?;
        let mae_value = abs_diff
            .mean_all()
            .map_err(|e| TurboCALMError::Candle(e))?
            .to_scalar::<f32>()
            .map_err(|e| TurboCALMError::Candle(e))?;

        Ok(mae_value)
    }

    /// Compute all similarity metrics at once
    pub fn all_metrics(a: &Tensor, b: &Tensor) -> Result<MetricsBundle> {
        Ok(MetricsBundle {
            cosine_similarity: Self::cosine_similarity(a, b)?,
            mse: Self::mse(a, b)?,
            rmse: Self::rmse(a, b)?,
            mae: Self::mae(a, b)?,
        })
    }
}

/// Bundle of similarity metrics
#[derive(Debug, Clone, PartialEq)]
pub struct MetricsBundle {
    pub cosine_similarity: f32,
    pub mse: f32,
    pub rmse: f32,
    pub mae: f32,
}

impl MetricsBundle {
    /// Check if the metrics indicate very similar tensors
    pub fn is_similar(&self, cosine_threshold: f32, mse_threshold: f32) -> bool {
        self.cosine_similarity > cosine_threshold && self.mse < mse_threshold
    }

    /// Get a summary string of all metrics
    pub fn summary(&self) -> String {
        format!(
            "cos_sim: {:.4}, mse: {:.6}, rmse: {:.6}, mae: {:.6}",
            self.cosine_similarity, self.mse, self.rmse, self.mae
        )
    }
}

/// Memory usage reporting utilities
pub struct MemoryReporter;

impl MemoryReporter {
    /// Get current memory usage statistics
    pub fn current_memory_usage() -> MemoryUsage {
        Self::current_memory_usage_for_device(None)
    }

    /// Get current memory usage statistics for a specific execution device.
    pub fn current_memory_usage_for_device(device: Option<&Device>) -> MemoryUsage {
        let (current_usage, source) = if device.is_some_and(Device::is_metal) {
            match Self::get_metal_allocated_mb() {
                Some(current_usage) => (current_usage, MemorySource::MetalAllocated),
                None => (Self::get_current_rss_mb(), MemorySource::CpuRss),
            }
        } else {
            (Self::get_current_rss_mb(), MemorySource::CpuRss)
        };

        MemoryUsage {
            current_mb: current_usage,
            peak_mb: current_usage,
            available_mb: Self::get_available_memory_mb(),
            source,
        }
    }

    /// Report memory usage for a tensor
    pub fn tensor_memory_usage(tensor: &Tensor) -> TensorMemoryInfo {
        let element_count = tensor.elem_count();
        let dtype_size = Self::dtype_size_bytes(tensor.dtype());
        let total_bytes = element_count * dtype_size;

        TensorMemoryInfo {
            shape: tensor.shape().clone(),
            dtype: tensor.dtype(),
            element_count,
            size_bytes: total_bytes,
            size_mb: total_bytes as f64 / (1024.0 * 1024.0),
            device: tensor.device().clone(),
        }
    }

    /// Get size in bytes for different data types
    fn dtype_size_bytes(dtype: candle_core::DType) -> usize {
        match dtype {
            candle_core::DType::U8 => 1,
            candle_core::DType::U32 => 4,
            candle_core::DType::I64 => 8,
            candle_core::DType::BF16 => 2,
            candle_core::DType::F16 => 2,
            candle_core::DType::F32 => 4,
            candle_core::DType::F64 => 8,
        }
    }

    /// Get current memory usage in MB (platform-specific implementation)
    fn get_current_rss_mb() -> f64 {
        #[cfg(target_os = "macos")]
        {
            Self::get_macos_rss_mb()
        }
        #[cfg(target_os = "linux")]
        {
            Self::get_linux_rss_mb()
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            0.0 // Fallback for unsupported platforms
        }
    }

    #[cfg(target_os = "macos")]
    fn get_metal_allocated_mb() -> Option<f64> {
        let device = metal::Device::system_default()?;
        Some(device.current_allocated_size() as f64 / (1024.0 * 1024.0))
    }

    #[cfg(not(target_os = "macos"))]
    fn get_metal_allocated_mb() -> Option<f64> {
        None
    }

    /// Get available memory in MB
    fn get_available_memory_mb() -> f64 {
        #[cfg(target_os = "macos")]
        {
            Self::get_macos_available_memory()
        }
        #[cfg(target_os = "linux")]
        {
            Self::get_linux_available_memory()
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            1024.0 // Fallback
        }
    }

    #[cfg(target_os = "macos")]
    fn get_macos_rss_mb() -> f64 {
        use std::process::Command;

        // Use ps to get memory usage for current process
        let pid = std::process::id().to_string();

        // Try multiple approaches for robustness

        // Approach 1: Use ps with RSS
        if let Ok(output) = Command::new("ps")
            .args(&["-o", "rss=", "-p", &pid])  // Use rss= to get only the value without header
            .output()
        {
            if let Ok(output_str) = String::from_utf8(output.stdout) {
                if let Some(rss_str) = output_str.lines().next() {
                    if let Ok(rss_kb) = rss_str.trim().parse::<f64>() {
                        if rss_kb > 0.0 {
                            return rss_kb / 1024.0; // Convert KB to MB
                        }
                    }
                }
            }
        }

        // Approach 2: Fallback to ps with VSZ (virtual memory size) if RSS fails
        if let Ok(output) = Command::new("ps")
            .args(&["-o", "vsz=", "-p", &pid])
            .output()
        {
            if let Ok(output_str) = String::from_utf8(output.stdout) {
                if let Some(vsz_str) = output_str.lines().next() {
                    if let Ok(vsz_kb) = vsz_str.trim().parse::<f64>() {
                        if vsz_kb > 0.0 {
                            // VSZ is usually higher than RSS, so use a conservative estimate
                            return (vsz_kb * 0.5) / 1024.0; // Convert KB to MB with conservative factor
                        }
                    }
                }
            }
        }

        0.0
    }

    #[cfg(target_os = "macos")]
    fn get_macos_available_memory() -> f64 {
        use std::process::Command;

        if let Ok(output) = Command::new("vm_stat").output() {
            if let Ok(output_str) = String::from_utf8(output.stdout) {
                let lines: Vec<&str> = output_str.lines().collect();

                // Parse page size from the first line
                // Example: "Mach Virtual Memory Statistics: (page size of 16384 bytes)"
                let page_size = if let Some(first_line) = lines.first() {
                    if let Some(start) = first_line.find("page size of ") {
                        let size_str = &first_line[start + 13..];
                        if let Some(end) = size_str.find(" bytes") {
                            size_str[..end].parse::<u64>().unwrap_or(16384)
                        } else {
                            16384
                        }
                    } else {
                        16384
                    }
                } else {
                    16384
                };

                let mut free_pages = 0u64;
                let mut inactive_pages = 0u64;

                // Parse the output to extract free and inactive pages
                for line in lines {
                    if line.starts_with("Pages free:") {
                        if let Some(value_str) = line.split(':').nth(1) {
                            let clean_str = value_str.trim().trim_end_matches('.');
                            free_pages = clean_str.parse::<u64>().unwrap_or(0);
                        }
                    } else if line.starts_with("Pages inactive:") {
                        if let Some(value_str) = line.split(':').nth(1) {
                            let clean_str = value_str.trim().trim_end_matches('.');
                            inactive_pages = clean_str.parse::<u64>().unwrap_or(0);
                        }
                    }
                }

                // Calculate available memory: (free + inactive) * page_size converted to MB
                let available_bytes = (free_pages + inactive_pages) * page_size;
                return available_bytes as f64 / (1024.0 * 1024.0);
            }
        }
        1024.0 // Fallback if parsing fails
    }

    #[cfg(target_os = "linux")]
    fn get_linux_rss_mb() -> f64 {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    if let Some(mem_str) = line.split_whitespace().nth(1) {
                        if let Ok(mem_kb) = mem_str.parse::<f64>() {
                            return mem_kb / 1024.0; // Convert KB to MB
                        }
                    }
                }
            }
        }
        0.0
    }

    #[cfg(target_os = "linux")]
    fn get_linux_available_memory() -> f64 {
        if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
            for line in meminfo.lines() {
                if line.starts_with("MemAvailable:") {
                    if let Some(mem_str) = line.split_whitespace().nth(1) {
                        if let Ok(mem_kb) = mem_str.parse::<f64>() {
                            return mem_kb / 1024.0; // Convert KB to MB
                        }
                    }
                }
            }
        }
        1024.0
    }

    /// Log memory usage with info level
    pub fn log_memory_usage() {
        let usage = Self::current_memory_usage();
        info!(
            "Memory ({:?}): current {:.1} MB, peak {:.1} MB, available {:.1} MB",
            usage.source, usage.current_mb, usage.peak_mb, usage.available_mb
        );
    }

    /// Log tensor memory info
    pub fn log_tensor_memory(name: &str, tensor: &Tensor) {
        let info = Self::tensor_memory_usage(tensor);
        debug!(
            "Tensor '{}': {:?}, {:.1} MB on {:?}",
            name, info.shape, info.size_mb, info.device
        );
    }
}

/// Memory usage information
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryUsage {
    pub current_mb: f64,
    pub peak_mb: f64,
    pub available_mb: f64,
    pub source: MemorySource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemorySource {
    MetalAllocated,
    CpuRss,
}

impl MemoryUsage {
    /// Check if memory usage is high (> 80% of available)
    pub fn is_high_usage(&self) -> bool {
        if self.available_mb > 0.0 {
            (self.current_mb / (self.current_mb + self.available_mb)) > 0.8
        } else {
            false
        }
    }

    /// Get memory usage percentage
    pub fn usage_percentage(&self) -> f64 {
        if self.available_mb > 0.0 {
            (self.current_mb / (self.current_mb + self.available_mb)) * 100.0
        } else {
            0.0
        }
    }
}

/// Information about tensor memory usage
#[derive(Debug, Clone)]
pub struct TensorMemoryInfo {
    pub shape: candle_core::Shape,
    pub dtype: candle_core::DType,
    pub element_count: usize,
    pub size_bytes: usize,
    pub size_mb: f64,
    pub device: Device,
}

impl TensorMemoryInfo {
    /// Check if tensor is large (> 100 MB)
    pub fn is_large(&self) -> bool {
        self.size_mb > 100.0
    }

    /// Get formatted size string
    pub fn size_string(&self) -> String {
        if self.size_mb < 1.0 {
            format!("{:.0} KB", self.size_bytes as f64 / 1024.0)
        } else if self.size_mb < 1024.0 {
            format!("{:.1} MB", self.size_mb)
        } else {
            format!("{:.1} GB", self.size_mb / 1024.0)
        }
    }
}

/// Metrics tracking for model training/evaluation
#[derive(Debug, Clone)]
pub struct MetricsTracker {
    metrics: HashMap<String, Vec<f32>>,
}

impl MetricsTracker {
    /// Create a new metrics tracker
    pub fn new() -> Self {
        Self {
            metrics: HashMap::new(),
        }
    }

    /// Record a metric value
    pub fn record(&mut self, name: &str, value: f32) {
        self.metrics
            .entry(name.to_string())
            .or_insert_with(Vec::new)
            .push(value);
    }

    /// Get the latest value for a metric
    pub fn latest(&self, name: &str) -> Option<f32> {
        self.metrics
            .get(name)
            .and_then(|values| values.last().copied())
    }

    /// Get the average value for a metric
    pub fn average(&self, name: &str) -> Option<f32> {
        self.metrics.get(name).map(|values| {
            if values.is_empty() {
                0.0
            } else {
                values.iter().sum::<f32>() / values.len() as f32
            }
        })
    }

    /// Get all recorded values for a metric
    pub fn values(&self, name: &str) -> Option<&[f32]> {
        self.metrics.get(name).map(|v| v.as_slice())
    }

    /// Get all metric names
    pub fn metric_names(&self) -> Vec<String> {
        self.metrics.keys().cloned().collect()
    }

    /// Clear all metrics
    pub fn clear(&mut self) {
        self.metrics.clear();
    }

    /// Log summary of all metrics
    pub fn log_summary(&self) {
        info!("Metrics Summary:");
        for (name, values) in &self.metrics {
            if let (Some(latest), Some(avg)) = (values.last(), self.average(name)) {
                info!(
                    "  {}: latest={:.4}, avg={:.4}, count={}",
                    name,
                    latest,
                    avg,
                    values.len()
                );
            }
        }
    }
}

impl Default for MetricsTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Shape, Tensor};

    fn create_test_tensor(values: &[f32]) -> Result<Tensor> {
        Ok(Tensor::from_slice(values, (values.len(),), &Device::Cpu)?)
    }

    #[test]
    fn test_cosine_similarity() -> Result<()> {
        let a = create_test_tensor(&[1.0, 0.0, 0.0])?;
        let b = create_test_tensor(&[1.0, 0.0, 0.0])?;
        let c = create_test_tensor(&[0.0, 1.0, 0.0])?;

        // Identical vectors should have cosine similarity of 1.0
        let sim_aa = SimilarityMetrics::cosine_similarity(&a, &b)?;
        assert!((sim_aa - 1.0).abs() < 1e-6);

        // Orthogonal vectors should have cosine similarity of 0.0
        let sim_ab = SimilarityMetrics::cosine_similarity(&a, &c)?;
        assert!(sim_ab.abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_mse() -> Result<()> {
        let a = create_test_tensor(&[1.0, 2.0, 3.0])?;
        let b = create_test_tensor(&[1.0, 2.0, 3.0])?;
        let c = create_test_tensor(&[2.0, 3.0, 4.0])?;

        // Identical tensors should have MSE of 0.0
        let mse_aa = SimilarityMetrics::mse(&a, &b)?;
        assert!(mse_aa.abs() < 1e-6);

        // Different tensors should have non-zero MSE
        let mse_ac = SimilarityMetrics::mse(&a, &c)?;
        assert_eq!(mse_ac, 1.0); // Each element differs by 1, so MSE = (1²+1²+1²)/3 = 1

        Ok(())
    }

    #[test]
    fn test_metrics_bundle() -> Result<()> {
        let a = create_test_tensor(&[1.0, 2.0, 3.0])?;
        let b = create_test_tensor(&[1.1, 2.1, 3.1])?;

        let metrics = SimilarityMetrics::all_metrics(&a, &b)?;
        assert!(metrics.cosine_similarity > 0.99); // Very similar vectors
        assert!(metrics.mse > 0.0 && metrics.mse < 0.1); // Small but non-zero error
        assert!(metrics.rmse > 0.0 && metrics.rmse < 0.5);
        assert!(metrics.mae > 0.0 && metrics.mae < 0.2);

        assert!(metrics.is_similar(0.95, 0.1));

        Ok(())
    }

    #[test]
    fn test_tensor_memory_info() {
        let tensor = Tensor::zeros((100, 100), DType::F32, &Device::Cpu).unwrap();
        let info = MemoryReporter::tensor_memory_usage(&tensor);

        assert_eq!(info.element_count, 10000);
        assert_eq!(info.size_bytes, 40000); // 10000 * 4 bytes per f32
        assert!((info.size_mb - 0.038).abs() < 0.001); // ~0.038 MB

        assert!(!info.is_large()); // < 100 MB
    }

    #[test]
    fn test_memory_usage() {
        let usage = MemoryUsage {
            current_mb: 500.0,
            peak_mb: 600.0,
            available_mb: 1500.0,
            source: MemorySource::CpuRss,
        };

        assert!(!usage.is_high_usage()); // 500/(500+1500) = 25% < 80%
        assert!((usage.usage_percentage() - 25.0).abs() < 1.0);
    }

    #[test]
    fn test_memory_usage_reports_cpu_rss_for_cpu_device() {
        let usage = MemoryReporter::current_memory_usage_for_device(Some(&Device::Cpu));
        assert_eq!(usage.source, MemorySource::CpuRss);
        assert!(usage.current_mb >= 0.0);
    }

    #[test]
    fn test_metrics_tracker() {
        let mut tracker = MetricsTracker::new();

        tracker.record("loss", 1.0);
        tracker.record("loss", 0.8);
        tracker.record("loss", 0.6);

        assert_eq!(tracker.latest("loss"), Some(0.6));
        assert!((tracker.average("loss").unwrap() - 0.8).abs() < 1e-6);
        assert_eq!(tracker.values("loss").unwrap().len(), 3);

        tracker.record("accuracy", 0.9);
        assert_eq!(tracker.metric_names().len(), 2);
    }
}
