use crate::cache::KvCache;
use candle_core::{Result, Tensor};

pub struct DenseKvCache {
    keys: Vec<Tensor>,
    values: Vec<Tensor>,
}

impl Default for DenseKvCache {
    fn default() -> Self {
        Self::new()
    }
}

impl DenseKvCache {
    pub fn new() -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
        }
    }
}

impl KvCache for DenseKvCache {
    fn append(&mut self, key: &Tensor, value: &Tensor) -> Result<()> {
        self.keys.push(key.clone());
        self.values.push(value.clone());
        Ok(())
    }

    fn get_key(&mut self) -> Result<Tensor> {
        if self.keys.is_empty() {
            return Err(candle_core::Error::Msg("KV Cache is empty".to_string()));
        }
        let rank = self.keys[0].dims().len();
        if rank < 1 {
            return Err(candle_core::Error::Msg("Invalid tensor rank".to_string()));
        }
        let cat_dim = if rank >= 2 { rank - 2 } else { 0 };
        Tensor::cat(&self.keys, cat_dim)
    }

    fn get_value(&mut self) -> Result<Tensor> {
        if self.values.is_empty() {
            return Err(candle_core::Error::Msg("KV Cache is empty".to_string()));
        }
        let rank = self.values[0].dims().len();
        let cat_dim = if rank >= 2 { rank - 2 } else { 0 };
        Tensor::cat(&self.values, cat_dim)
    }

    fn clear(&mut self) {
        self.keys.clear();
        self.values.clear();
    }
}
