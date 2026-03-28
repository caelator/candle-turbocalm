use crate::cache::KvCache;
use anyhow::Result;
use candle_core::Tensor;

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
            return Err(anyhow::Error::msg("KV Cache is empty"));
        }
        let rank = self.keys[0].dims().len();
        if rank < 1 {
            return Err(anyhow::Error::msg("Invalid tensor rank"));
        }
        let cat_dim = if rank >= 2 { rank - 2 } else { 0 };
        Ok(Tensor::cat(&self.keys, cat_dim).map_err(anyhow::Error::from)?)
    }

    fn get_value(&mut self) -> Result<Tensor> {
        if self.values.is_empty() {
            return Err(anyhow::Error::msg("KV Cache is empty"));
        }
        let rank = self.values[0].dims().len();
        let cat_dim = if rank >= 2 { rank - 2 } else { 0 };
        Ok(Tensor::cat(&self.values, cat_dim).map_err(anyhow::Error::from)?)
    }

    fn clear(&mut self) {
        self.keys.clear();
        self.values.clear();
    }
}
