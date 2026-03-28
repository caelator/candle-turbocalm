use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantProfile {
    pub bit_width: u8,
    pub rotation_seed: u64,
    pub qjl_dim: usize,
    pub qjl_threshold: f32,
    pub scale_mode: String,
}

impl Default for QuantProfile {
    fn default() -> Self {
        Self {
            bit_width: 8,
            rotation_seed: 42,
            qjl_dim: 16,
            qjl_threshold: 0.0,
            scale_mode: "per_token".to_string(),
        }
    }
}
