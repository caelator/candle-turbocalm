//! Multi-objective fitness function for quantization parameter optimization
//!
//! Implements the fitness function: memory_gain - λ1·ΔBrierLM - λ2·cosine_penalty - λ3·latency_penalty

use crate::{FitnessMetrics, ObjectiveWeights, QuantProfile};
use crate::dataset::{ProcessedDataset, TensorStats};
use anyhow::Result;
use candle_core::{Device, Tensor};
use std::time::Instant;

/// Objective function evaluator
pub struct ObjectiveFunction {
    /// Weights for multi-objective combination
    pub weights: ObjectiveWeights,
    /// Reference (unquantized) metrics for comparison
    pub reference_metrics: Option<ReferenceMetrics>,
}

/// Reference metrics from unquantized model
#[derive(Debug, Clone)]
pub struct ReferenceMetrics {
    /// Original memory usage (bytes)
    pub memory_usage: usize,
    /// Baseline Brier score for language modeling
    pub baseline_brier_lm: f64,
    /// Baseline inference latency (ms)
    pub baseline_latency_ms: f64,
    /// Reference KV cache statistics
    pub reference_kv_stats: Vec<TensorStats>,
}

/// Quantization evaluation results
#[derive(Debug, Clone)]
pub struct QuantizationResult {
    /// Quantized model memory usage (bytes)
    pub quantized_memory: usize,
    /// Brier score with quantization
    pub quantized_brier_lm: f64,
    /// Inference latency with quantization (ms)
    pub quantized_latency_ms: f64,
    /// Cosine similarity between original and quantized activations
    pub cosine_similarity: f64,
    /// Additional metrics for analysis
    pub metrics: FitnessMetrics,
}

impl ObjectiveFunction {
    /// Create new objective function evaluator
    pub fn new(weights: ObjectiveWeights) -> Self {
        Self {
            weights,
            reference_metrics: None,
        }
    }

    /// Set reference metrics from unquantized evaluation
    pub fn set_reference(&mut self, reference: ReferenceMetrics) {
        self.reference_metrics = Some(reference);
    }

    /// Evaluate a quantization profile
    pub fn evaluate(
        &self,
        profile: &QuantProfile,
        dataset: &ProcessedDataset,
        device: &Device,
    ) -> Result<(FitnessMetrics, f64)> {
        let reference = self.reference_metrics
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Reference metrics not set"))?;

        // Simulate quantized model evaluation
        let quant_result = self.simulate_quantization(profile, dataset, device)?;

        // Calculate individual fitness components
        let memory_gain = self.calculate_memory_gain(reference, &quant_result);
        let delta_brier_lm = self.calculate_brier_delta(reference, &quant_result);
        let cosine_penalty = self.calculate_cosine_penalty(&quant_result);
        let latency_penalty = self.calculate_latency_penalty(reference, &quant_result);

        let fitness = FitnessMetrics {
            memory_gain,
            delta_brier_lm,
            cosine_penalty,
            latency_penalty,
        };

        // Calculate weighted objective value (to minimize)
        let objective_value = -memory_gain
            + self.weights.lambda1 * delta_brier_lm
            + self.weights.lambda2 * cosine_penalty
            + self.weights.lambda3 * latency_penalty;

        Ok((fitness, objective_value))
    }

    /// Simulate quantization evaluation (placeholder for actual model inference)
    fn simulate_quantization(
        &self,
        profile: &QuantProfile,
        dataset: &ProcessedDataset,
        _device: &Device,
    ) -> Result<QuantizationResult> {
        let reference = self.reference_metrics.as_ref().unwrap();

        // Legitimate Phase 5 placeholder: memory can be estimated analytically from
        // the quantization configuration without running a model forward pass, even
        // though exact accounting should eventually use real layer metadata.
        let bits_reduction_factor = 32.0 / profile.bit_width as f64;
        let memory_reduction = 0.7; // Assuming 70% of memory is quantizable
        let quantized_memory = (reference.memory_usage as f64 *
                               (1.0 - memory_reduction + memory_reduction / bits_reduction_factor)) as usize;

        // TODO(Phase 5): replace synthetic quality, latency, and activation-similarity
        // estimates with real baseline-vs-quantized model inference over `dataset`.
        let quality_degradation = self.estimate_quality_degradation(profile);
        let quantized_brier_lm = reference.baseline_brier_lm * (1.0 + quality_degradation);

        // Simulate latency impact (lower precision can be faster, but QJL adds overhead)
        let qjl_overhead = (profile.qjl_dim as f64).ln() * 0.01; // Log scaling for QJL overhead
        let precision_speedup = match profile.bit_width {
            2 => 0.3, // 30% faster for 2-bit
            3 => 0.15, // 15% faster for 3-bit
            4 => 0.05, // 5% faster for 4-bit
            _ => 0.0,
        };
        let latency_factor = 1.0 - precision_speedup + qjl_overhead;
        let quantized_latency_ms = reference.baseline_latency_ms * latency_factor;

        // Simulate cosine similarity based on quantization parameters
        let cosine_similarity = self.estimate_cosine_similarity(profile, dataset)?;

        // Create fitness metrics
        let memory_gain = (reference.memory_usage - quantized_memory) as f64 / reference.memory_usage as f64;
        let delta_brier_lm = quantized_brier_lm - reference.baseline_brier_lm;
        let cosine_penalty = 1.0 - cosine_similarity;
        let latency_penalty = (quantized_latency_ms - reference.baseline_latency_ms) / reference.baseline_latency_ms;

        let fitness = FitnessMetrics {
            memory_gain,
            delta_brier_lm,
            cosine_penalty,
            latency_penalty: latency_penalty.max(0.0), // No negative penalty for speedup
        };

        Ok(QuantizationResult {
            quantized_memory,
            quantized_brier_lm,
            quantized_latency_ms,
            cosine_similarity,
            metrics: fitness,
        })
    }

    /// Estimate quality degradation from quantization parameters
    fn estimate_quality_degradation(&self, profile: &QuantProfile) -> f64 {
        // TODO(Phase 5): compute this from real output quality metrics instead of
        // the current heuristic once quantized model execution is wired up.
        // Base degradation from bit width
        let bit_degradation = match profile.bit_width {
            2 => 0.15, // 15% degradation for 2-bit
            3 => 0.08, // 8% degradation for 3-bit
            4 => 0.03, // 3% degradation for 4-bit
            _ => 0.01,
        };

        // QJL dimension impact (lower dimension = more degradation)
        let qjl_factor = 1.0 - (profile.qjl_dim as f64 / 64.0).min(1.0) * 0.3;

        // Continuous parameter impacts
        let clipping_impact = (1.0 - profile.continuous.clipping_percentile) * 0.5;
        let scale_impact = (profile.continuous.scale_multiplier - 1.0).abs() * 0.1;
        let threshold_impact = (profile.continuous.qjl_threshold - 1e-4).abs() * 1000.0;

        bit_degradation * qjl_factor + clipping_impact + scale_impact + threshold_impact
    }

    /// Estimate cosine similarity between original and quantized activations
    fn estimate_cosine_similarity(&self, profile: &QuantProfile, _dataset: &ProcessedDataset) -> Result<f64> {
        // TODO(Phase 5): compare real activation tensors from the baseline and
        // quantized model instead of using this hand-tuned similarity proxy.
        // Simulate activation comparison based on quantization parameters
        let base_similarity = match profile.bit_width {
            2 => 0.85, // Lower similarity for aggressive quantization
            3 => 0.92,
            4 => 0.97,
            _ => 0.99,
        };

        // QJL and continuous parameter adjustments
        let qjl_boost = (profile.qjl_dim as f64 / 64.0).min(1.0) * 0.05;
        let clipping_penalty = (1.0 - profile.continuous.clipping_percentile) * 0.1;

        let similarity = (base_similarity + qjl_boost - clipping_penalty)
            .max(0.0)
            .min(1.0);

        Ok(similarity)
    }

    /// Calculate memory gain component
    fn calculate_memory_gain(&self, _reference: &ReferenceMetrics, result: &QuantizationResult) -> f64 {
        result.metrics.memory_gain
    }

    /// Calculate Brier score delta component
    fn calculate_brier_delta(&self, _reference: &ReferenceMetrics, result: &QuantizationResult) -> f64 {
        result.metrics.delta_brier_lm
    }

    /// Calculate cosine similarity penalty
    fn calculate_cosine_penalty(&self, result: &QuantizationResult) -> f64 {
        result.metrics.cosine_penalty
    }

    /// Calculate latency penalty component
    fn calculate_latency_penalty(&self, _reference: &ReferenceMetrics, result: &QuantizationResult) -> f64 {
        result.metrics.latency_penalty
    }
}

/// Batch evaluation for multiple profiles
pub struct BatchEvaluator {
    objective_fn: ObjectiveFunction,
}

impl BatchEvaluator {
    /// Create new batch evaluator
    pub fn new(weights: ObjectiveWeights) -> Self {
        Self {
            objective_fn: ObjectiveFunction::new(weights),
        }
    }

    /// Set reference metrics
    pub fn set_reference(&mut self, reference: ReferenceMetrics) {
        self.objective_fn.set_reference(reference);
    }

    /// Evaluate multiple profiles
    pub fn evaluate_batch(
        &self,
        profiles: &[QuantProfile],
        dataset: &ProcessedDataset,
        device: &Device,
    ) -> Result<Vec<(FitnessMetrics, f64)>> {
        profiles.iter()
            .map(|profile| self.objective_fn.evaluate(profile, dataset, device))
            .collect()
    }
}

/// Utility function to create reference metrics from baseline evaluation
pub fn create_reference_metrics(
    dataset: &ProcessedDataset,
    device: &Device,
) -> Result<ReferenceMetrics> {
    // TODO(Phase 5): baseline Brier score and latency must come from the real
    // unquantized model on the calibration corpus. The memory estimate can remain
    // analytical for now, but should eventually use actual loaded-model metadata.
    let baseline_memory = estimate_model_memory_usage(32); // 32-bit baseline
    let baseline_brier = simulate_brier_score(dataset, 32)?;
    let baseline_latency = simulate_inference_latency(dataset, device, 32)?;

    // Extract reference KV statistics
    let reference_kv_stats = dataset.kv_traces.iter()
        .map(|trace| trace.key_stats.clone())
        .collect();

    Ok(ReferenceMetrics {
        memory_usage: baseline_memory,
        baseline_brier_lm: baseline_brier,
        baseline_latency_ms: baseline_latency,
        reference_kv_stats,
    })
}

/// Estimate model memory usage based on precision
fn estimate_model_memory_usage(bits: u8) -> usize {
    // Legitimate Phase 5 placeholder: memory accounting does not require a forward
    // pass, but this fixed 7B assumption should be replaced by real model metadata.
    // Simplified estimation: assume model size scales with precision
    let base_size_gb = 7.0; // 7B parameter model
    let bytes_per_param = bits as f64 / 8.0;
    (base_size_gb * 1e9 * bytes_per_param) as usize
}

/// Simulate Brier score evaluation
fn simulate_brier_score(_dataset: &ProcessedDataset, bits: u8) -> Result<f64> {
    // TODO(Phase 5): replace the synthetic score with a real Brier evaluation from
    // model logits over the calibration dataset.
    let base_brier = 0.25; // Baseline Brier score
    let precision_factor = (32.0 / bits as f64 - 1.0) * 0.02; // 2% degradation per bit reduction
    Ok(base_brier * (1.0 + precision_factor))
}

/// Simulate inference latency
fn simulate_inference_latency(dataset: &ProcessedDataset, device: &Device, bits: u8) -> Result<f64> {
    // TODO(Phase 5): replace tensor-allocation timing with real end-to-end latency
    // measurements from the baseline model on representative calibration batches.
    let start = Instant::now();

    // Simulate some computation proportional to dataset size and precision
    let batch_size = dataset.len().min(32);
    let computation_factor = bits as f64 / 32.0;

    // Simulate tensor operations
    for i in 0..batch_size {
        let tensor_size = dataset.input_ids[i].len();
        let _dummy = Tensor::randn(0.0, 1.0 * computation_factor, tensor_size, device)?;
    }

    let duration = start.elapsed();
    Ok(duration.as_millis() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ContinuousParams};
    use crate::dataset::ProcessedDataset;
    use candle_core::Device;

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected:.12}, got {actual:.12}",
        );
    }

    fn create_test_dataset(device: &Device) -> ProcessedDataset {
        ProcessedDataset {
            input_ids: vec![vec![1, 2, 3, 4]],
            attention_masks: vec![vec![1, 1, 1, 1]],
            kv_traces: vec![],
            device: device.clone(),
        }
    }

    #[test]
    fn test_objective_evaluation() -> Result<()> {
        let weights = ObjectiveWeights::default();
        let mut objective = ObjectiveFunction::new(weights);
        let device = Device::Cpu;
        let dataset = create_test_dataset(&device);

        // Create mock reference metrics
        let reference = ReferenceMetrics {
            memory_usage: 1_000_000_000, // 1GB
            baseline_brier_lm: 0.25,
            baseline_latency_ms: 100.0,
            reference_kv_stats: vec![],
        };
        objective.set_reference(reference);

        let profile = QuantProfile {
            bit_width: 4,
            qjl_dim: 32,
            rotation_seed: 42,
            continuous: ContinuousParams::default(),
        };

        let (fitness, objective_value) = objective.evaluate(&profile, &dataset, &device)?;

        assert_close(fitness.memory_gain, 0.6125, 1e-12);
        assert_close(fitness.delta_brier_lm, 0.012625, 1e-12);
        assert_close(fitness.cosine_penalty, 0.01, 1e-12);
        assert_eq!(fitness.latency_penalty, 0.0);
        assert_close(objective_value, -0.594875, 1e-12);

        Ok(())
    }

    #[test]
    fn test_quality_degradation_estimation() {
        let weights = ObjectiveWeights::default();
        let objective = ObjectiveFunction::new(weights);

        let profile_2bit = QuantProfile {
            bit_width: 2,
            qjl_dim: 16,
            rotation_seed: 42,
            continuous: ContinuousParams::default(),
        };

        let profile_4bit = QuantProfile {
            bit_width: 4,
            qjl_dim: 64,
            rotation_seed: 42,
            continuous: ContinuousParams::default(),
        };

        let degradation_2bit = objective.estimate_quality_degradation(&profile_2bit);
        let degradation_4bit = objective.estimate_quality_degradation(&profile_4bit);

        // 2-bit should have higher degradation than 4-bit
        assert!(degradation_2bit > degradation_4bit);
        assert!(degradation_2bit > 0.0);
        assert!(degradation_4bit > 0.0);
    }
}
