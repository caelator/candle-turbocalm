//! Dataset loading and preprocessing for calibration
//!
//! Handles loading calibration corpus from JSONL format, tokenization,
//! and generation of KV cache trace tensors for optimization.

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use tokenizers::Tokenizer;

/// A single calibration sample
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationSample {
    /// Input text for calibration
    pub text: String,
    /// Optional metadata
    pub metadata: Option<serde_json::Value>,
}

/// Dataset of calibration samples
#[derive(Debug, Clone)]
pub struct CalibrationDataset {
    pub samples: Vec<CalibrationSample>,
}

/// Tokenized and processed dataset ready for model evaluation
#[derive(Debug)]
pub struct ProcessedDataset {
    /// Input token sequences
    pub input_ids: Vec<Vec<u32>>,
    /// Attention masks
    pub attention_masks: Vec<Vec<u32>>,
    /// KV cache trace tensors (for memory profiling)
    pub kv_traces: Vec<KvTrace>,
    /// Device for tensor operations
    pub device: Device,
}

/// KV cache trace information
#[derive(Debug, Clone)]
pub struct KvTrace {
    /// Sequence length
    pub seq_len: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Key tensor shape and statistics
    pub key_stats: TensorStats,
    /// Value tensor shape and statistics
    pub value_stats: TensorStats,
}

/// Statistics for tensor analysis
#[derive(Debug, Clone)]
pub struct TensorStats {
    /// Mean absolute value
    pub mean_abs: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Min value
    pub min_val: f64,
    /// Max value
    pub max_val: f64,
    /// L2 norm
    pub l2_norm: f64,
}

impl CalibrationDataset {
    /// Load dataset from JSONL file
    pub fn from_jsonl<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(&path)
            .with_context(|| format!("Failed to open dataset file: {:?}", path.as_ref()))?;

        let reader = BufReader::new(file);
        let mut samples = Vec::new();

        for (line_num, line) in reader.lines().enumerate() {
            let line = line.with_context(|| format!("Failed to read line {}", line_num + 1))?;

            if line.trim().is_empty() {
                continue;
            }

            let sample: CalibrationSample = serde_json::from_str(&line)
                .with_context(|| format!("Failed to parse JSON on line {}", line_num + 1))?;

            samples.push(sample);
        }

        if samples.is_empty() {
            anyhow::bail!("Dataset is empty");
        }

        tracing::info!("Loaded {} samples from {:?}", samples.len(), path.as_ref());

        Ok(Self { samples })
    }

    /// Create a subset of the dataset
    pub fn subset(&self, max_samples: Option<usize>) -> Self {
        let samples = match max_samples {
            Some(max) if max < self.samples.len() => self.samples[..max].to_vec(),
            _ => self.samples.clone(),
        };

        Self { samples }
    }

    /// Get number of samples
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Check if dataset is empty
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

impl ProcessedDataset {
    /// Process dataset for model evaluation
    pub fn from_dataset(
        dataset: &CalibrationDataset,
        tokenizer: &Tokenizer,
        device: &Device,
        max_length: Option<usize>,
    ) -> Result<Self> {
        let max_len = max_length.unwrap_or(512);
        let mut input_ids = Vec::new();
        let mut attention_masks = Vec::new();
        let mut kv_traces = Vec::new();

        for sample in &dataset.samples {
            // Tokenize the text
            let encoding = tokenizer
                .encode(sample.text.as_str(), false)
                .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

            let mut tokens = encoding.get_ids().to_vec();
            let _original_len = tokens.len();

            // Truncate or pad to max_length
            if tokens.len() > max_len {
                tokens.truncate(max_len);
            }

            let seq_len = tokens.len();
            let mut mask = vec![1u32; seq_len];

            // Pad if necessary
            while tokens.len() < max_len {
                tokens.push(0); // PAD token
                mask.push(0); // PAD mask
            }

            input_ids.push(tokens);
            attention_masks.push(mask);

            // Generate synthetic KV trace (in practice, this would come from model execution)
            let kv_trace = Self::generate_kv_trace(seq_len, device)?;
            kv_traces.push(kv_trace);
        }

        Ok(Self {
            input_ids,
            attention_masks,
            kv_traces,
            device: device.clone(),
        })
    }

    /// Generate KV cache trace for a sequence (placeholder implementation)
    fn generate_kv_trace(seq_len: usize, device: &Device) -> Result<KvTrace> {
        // Standard transformer architecture parameters
        let num_heads = 32;
        let head_dim = 128;

        // Generate synthetic key and value tensors (in practice, these come from model)
        let key_tensor = Tensor::randn(0.0, 1.0, (seq_len, num_heads * head_dim), device)?;
        let value_tensor = Tensor::randn(0.0, 1.0, (seq_len, num_heads * head_dim), device)?;

        let key_stats = Self::compute_tensor_stats(&key_tensor)?;
        let value_stats = Self::compute_tensor_stats(&value_tensor)?;

        Ok(KvTrace {
            seq_len,
            num_heads,
            head_dim,
            key_stats,
            value_stats,
        })
    }

    /// Compute tensor statistics
    fn compute_tensor_stats(tensor: &Tensor) -> Result<TensorStats> {
        // Convert to CPU for statistical analysis
        let tensor_f32 = tensor.to_dtype(candle_core::DType::F32)?;
        let values: Vec<f32> = tensor_f32.flatten_all()?.to_vec1()?;

        let n = values.len() as f64;
        let sum: f64 = values.iter().map(|&x| x as f64).sum();
        let mean = sum / n;

        let sum_abs: f64 = values.iter().map(|&x| (x as f64).abs()).sum();
        let mean_abs = sum_abs / n;

        let variance: f64 = values
            .iter()
            .map(|&x| (x as f64 - mean).powi(2))
            .sum::<f64>()
            / n;
        let std_dev = variance.sqrt();

        let min_val = values
            .iter()
            .map(|&x| x as f64)
            .fold(f64::INFINITY, f64::min);
        let max_val = values
            .iter()
            .map(|&x| x as f64)
            .fold(f64::NEG_INFINITY, f64::max);

        let l2_norm = values
            .iter()
            .map(|&x| (x as f64).powi(2))
            .sum::<f64>()
            .sqrt();

        Ok(TensorStats {
            mean_abs,
            std_dev,
            min_val,
            max_val,
            l2_norm,
        })
    }

    /// Get batch for evaluation
    pub fn get_batch(&self, batch_indices: &[usize]) -> Result<DataBatch> {
        if batch_indices.iter().any(|&i| i >= self.input_ids.len()) {
            anyhow::bail!("Batch index out of bounds");
        }

        let batch_size = batch_indices.len();
        let seq_len = self.input_ids[0].len();

        // Collect input_ids for the batch
        let mut batch_input_ids = Vec::with_capacity(batch_size * seq_len);
        let mut batch_attention_masks = Vec::with_capacity(batch_size * seq_len);
        let mut batch_kv_traces = Vec::with_capacity(batch_size);

        for &idx in batch_indices {
            batch_input_ids.extend_from_slice(&self.input_ids[idx]);
            batch_attention_masks.extend_from_slice(&self.attention_masks[idx]);
            batch_kv_traces.push(self.kv_traces[idx].clone());
        }

        // Create tensors
        let input_ids_tensor =
            Tensor::from_slice(&batch_input_ids, (batch_size, seq_len), &self.device)?;

        let attention_mask_tensor =
            Tensor::from_slice(&batch_attention_masks, (batch_size, seq_len), &self.device)?;

        Ok(DataBatch {
            input_ids: input_ids_tensor,
            attention_mask: attention_mask_tensor,
            kv_traces: batch_kv_traces,
        })
    }

    /// Get number of samples
    pub fn len(&self) -> usize {
        self.input_ids.len()
    }

    /// Check if dataset is empty
    pub fn is_empty(&self) -> bool {
        self.input_ids.is_empty()
    }
}

/// A batch of data ready for model evaluation
#[derive(Debug)]
pub struct DataBatch {
    pub input_ids: Tensor,
    pub attention_mask: Tensor,
    pub kv_traces: Vec<KvTrace>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_jsonl_loading() -> Result<()> {
        let mut temp_file = NamedTempFile::new()?;

        // Write test JSONL data
        writeln!(
            temp_file,
            r#"{{"text": "Hello world", "metadata": {{"source": "test"}}}}"#
        )?;
        writeln!(
            temp_file,
            r#"{{"text": "This is a test sentence for calibration."}}"#
        )?;
        writeln!(
            temp_file,
            r#"{{"text": "Another example with different content."}}"#
        )?;

        let dataset = CalibrationDataset::from_jsonl(temp_file.path())?;

        assert_eq!(dataset.len(), 3);
        assert_eq!(dataset.samples[0].text, "Hello world");
        assert!(dataset.samples[0].metadata.is_some());
        assert!(dataset.samples[1].metadata.is_none());

        Ok(())
    }

    #[test]
    fn test_dataset_subset() -> Result<()> {
        let samples = vec![
            CalibrationSample {
                text: "Sample 1".to_string(),
                metadata: None,
            },
            CalibrationSample {
                text: "Sample 2".to_string(),
                metadata: None,
            },
            CalibrationSample {
                text: "Sample 3".to_string(),
                metadata: None,
            },
        ];

        let dataset = CalibrationDataset { samples };
        let subset = dataset.subset(Some(2));

        assert_eq!(subset.len(), 2);
        assert_eq!(subset.samples[0].text, "Sample 1");
        assert_eq!(subset.samples[1].text, "Sample 2");

        Ok(())
    }
}
