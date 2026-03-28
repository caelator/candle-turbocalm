use serde::{Deserialize, Serialize};

/// Unified quantization profile with all fields from both implementations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantProfile {
    /// Quantization bit width
    pub bit_width: u8,
    /// Random rotation seed
    pub rotation_seed: u64,
    /// Quasi-Jordan-Lie dimension
    pub qjl_dim: usize,
    /// QJL threshold
    pub qjl_threshold: f32,
    /// Scale mode for quantization
    pub scale_mode: String,
    /// Clipping percentile (0.0 to 1.0) - from calibrate version
    pub clipping_percentile: f64,
    /// Scale multiplier (0.1 to 10.0) - from calibrate version
    pub scale_multiplier: f64,
}

/// Continuous parameters optimized by CMA-ES
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinuousParams {
    /// Clipping percentile (0.0 to 1.0)
    pub clipping_percentile: f64,
    /// Scale multiplier (0.1 to 10.0)
    pub scale_multiplier: f64,
    /// QJL threshold (1e-6 to 1e-2)
    pub qjl_threshold: f64,
}

impl Default for ContinuousParams {
    fn default() -> Self {
        Self {
            clipping_percentile: 0.99,
            scale_multiplier: 1.0,
            qjl_threshold: 1e-4,
        }
    }
}

impl QuantProfile {
    /// Create from the old kv version
    pub fn from_kv_profile(
        bit_width: u8,
        rotation_seed: u64,
        qjl_dim: usize,
        qjl_threshold: f32,
        scale_mode: String,
    ) -> Self {
        Self {
            bit_width,
            rotation_seed,
            qjl_dim,
            qjl_threshold,
            scale_mode,
            clipping_percentile: 0.99,
            scale_multiplier: 1.0,
        }
    }

    /// Create from the old calibrate version with continuous params
    pub fn from_calibrate_profile(
        bit_width: u8,
        qjl_dim: usize,
        rotation_seed: u64,
        continuous: ContinuousParams,
    ) -> Self {
        Self {
            bit_width,
            rotation_seed,
            qjl_dim,
            qjl_threshold: continuous.qjl_threshold as f32,
            scale_mode: "per_token".to_string(),
            clipping_percentile: continuous.clipping_percentile,
            scale_multiplier: continuous.scale_multiplier,
        }
    }

    /// Get continuous params for compatibility with calibrate
    pub fn continuous_params(&self) -> ContinuousParams {
        ContinuousParams {
            clipping_percentile: self.clipping_percentile,
            scale_multiplier: self.scale_multiplier,
            qjl_threshold: self.qjl_threshold as f64,
        }
    }
}

impl Default for QuantProfile {
    fn default() -> Self {
        Self {
            bit_width: 8,
            rotation_seed: 42,
            qjl_dim: 16,
            qjl_threshold: 0.0,
            scale_mode: "per_token".to_string(),
            clipping_percentile: 0.99,
            scale_multiplier: 1.0,
        }
    }
}
