pub mod dense;

use candle_core::{Result, Tensor};
use crate::quant::profile::QuantProfile;

pub trait KvCache {
    fn append(&mut self, key: &Tensor, value: &Tensor) -> Result<()>;
    fn get_key(&self) -> Result<Tensor>;
    fn get_value(&self) -> Result<Tensor>;
}

// TurboKvCache can be declared here or in a separate file (e.g. turbo.rs)
// For now, we will add a skeleton implementation that utilizes the QuantProfile
pub struct TurboKvCache {
    profile: QuantProfile,
    // Add quantized keys and values storage here
}

impl TurboKvCache {
    pub fn new(profile: QuantProfile) -> Self {
        Self { profile }
    }
}
