use anyhow::Result;
use candle_core::{Shape, Tensor};
use std::collections::HashMap;
use tracing::{debug, error, info, warn};
use turbocalm_core::{AutoencoderConfig, CALMConfig};

/// Shape verification utility for validating tensor shapes against model configurations
pub struct ShapeVerifier {
    expected_shapes: HashMap<String, ExpectedShape>,
    strict_mode: bool,
}

impl ShapeVerifier {
    /// Create a new shape verifier
    pub fn new(strict_mode: bool) -> Self {
        Self {
            expected_shapes: HashMap::new(),
            strict_mode,
        }
    }

    /// Create a shape verifier for CALM model
    pub fn for_calm_model(config: &CALMConfig, strict_mode: bool) -> Self {
        let mut verifier = Self::new(strict_mode);
        verifier.add_calm_expected_shapes(config);
        verifier
    }

    /// Create a shape verifier for autoencoder model
    pub fn for_autoencoder(config: &AutoencoderConfig, strict_mode: bool) -> Self {
        let mut verifier = Self::new(strict_mode);
        verifier.add_autoencoder_expected_shapes(config);
        verifier
    }

    /// Add expected shape for a tensor
    pub fn add_expected_shape(&mut self, name: &str, shape: ExpectedShape) {
        self.expected_shapes.insert(name.to_string(), shape);
    }

    /// Add expected shape with exact dimensions
    pub fn add_exact_shape(&mut self, name: &str, dims: Vec<usize>) {
        self.add_expected_shape(name, ExpectedShape::Exact(dims));
    }

    /// Add expected shape with constraints
    pub fn add_constrained_shape(&mut self, name: &str, constraints: ShapeConstraints) {
        self.add_expected_shape(name, ExpectedShape::Constrained(constraints));
    }

    /// Verify all tensors in a model
    pub fn verify_model_shapes(
        &self,
        tensors: &HashMap<String, Tensor>,
    ) -> Result<VerificationReport> {
        info!("Verifying shapes for {} tensors", tensors.len());

        let mut results = Vec::new();
        let mut error_count = 0;
        let mut warning_count = 0;

        for (name, tensor) in tensors {
            let result = self.verify_tensor_shape(name, tensor);
            match &result.status {
                VerificationStatus::Pass => {}
                VerificationStatus::Warning(_) => warning_count += 1,
                VerificationStatus::Error(_) => error_count += 1,
            }
            results.push(result);
        }

        // Check for missing expected tensors
        for expected_name in self.expected_shapes.keys() {
            if !tensors.contains_key(expected_name) {
                let result = TensorVerificationResult {
                    tensor_name: expected_name.clone(),
                    expected_shape: self.expected_shapes.get(expected_name).cloned(),
                    actual_shape: None,
                    status: VerificationStatus::Warning(format!(
                        "Expected tensor '{}' not found",
                        expected_name
                    )),
                };
                results.push(result);
                warning_count += 1;
            }
        }

        let report = VerificationReport {
            total_tensors: tensors.len(),
            results,
            error_count,
            warning_count,
            strict_mode: self.strict_mode,
        };

        if error_count > 0 && self.strict_mode {
            error!("Shape verification failed with {} errors", error_count);
            return Err(anyhow::anyhow!("Shape verification failed"));
        } else if error_count > 0 {
            warn!("Shape verification completed with {} errors", error_count);
        }

        if warning_count > 0 {
            warn!(
                "Shape verification completed with {} warnings",
                warning_count
            );
        }

        info!(
            "Shape verification complete: {} tensors, {} errors, {} warnings",
            tensors.len(),
            error_count,
            warning_count
        );

        Ok(report)
    }

    /// Verify shape of a single tensor
    pub fn verify_tensor_shape(&self, name: &str, tensor: &Tensor) -> TensorVerificationResult {
        debug!("Verifying shape for tensor: {}", name);

        let actual_shape = tensor.shape().clone();

        if let Some(expected) = self.expected_shapes.get(name) {
            let status = self.check_shape_match(expected, &actual_shape);
            TensorVerificationResult {
                tensor_name: name.to_string(),
                expected_shape: Some(expected.clone()),
                actual_shape: Some(actual_shape),
                status,
            }
        } else {
            // No explicit expectation - check if it matches common patterns
            let status = self.check_common_patterns(name, &actual_shape);
            TensorVerificationResult {
                tensor_name: name.to_string(),
                expected_shape: None,
                actual_shape: Some(actual_shape),
                status,
            }
        }
    }

    /// Check if actual shape matches expected shape
    fn check_shape_match(&self, expected: &ExpectedShape, actual: &Shape) -> VerificationStatus {
        match expected {
            ExpectedShape::Exact(expected_dims) => {
                let actual_dims: Vec<usize> = actual.dims().to_vec();
                if actual_dims == *expected_dims {
                    VerificationStatus::Pass
                } else {
                    VerificationStatus::Error(format!(
                        "Shape mismatch: expected {:?}, got {:?}",
                        expected_dims, actual_dims
                    ))
                }
            }
            ExpectedShape::Constrained(constraints) => self.check_constraints(constraints, actual),
            ExpectedShape::Pattern(pattern) => self.check_pattern(pattern, actual),
        }
    }

    /// Check shape constraints
    fn check_constraints(
        &self,
        constraints: &ShapeConstraints,
        actual: &Shape,
    ) -> VerificationStatus {
        let dims = actual.dims();

        // Check rank
        if let Some(expected_rank) = constraints.rank {
            if dims.len() != expected_rank {
                return VerificationStatus::Error(format!(
                    "Rank mismatch: expected {}, got {}",
                    expected_rank,
                    dims.len()
                ));
            }
        }

        // Check minimum dimensions
        if let Some(ref min_dims) = constraints.min_dims {
            if dims.len() < min_dims.len() {
                return VerificationStatus::Error(format!(
                    "Insufficient dimensions: expected at least {}, got {}",
                    min_dims.len(),
                    dims.len()
                ));
            }

            for (i, &min_dim) in min_dims.iter().enumerate() {
                if i < dims.len() && dims[i] < min_dim {
                    return VerificationStatus::Error(format!(
                        "Dimension {} too small: expected at least {}, got {}",
                        i, min_dim, dims[i]
                    ));
                }
            }
        }

        // Check maximum dimensions
        if let Some(ref max_dims) = constraints.max_dims {
            for (i, &max_dim) in max_dims.iter().enumerate() {
                if i < dims.len() && dims[i] > max_dim {
                    return VerificationStatus::Error(format!(
                        "Dimension {} too large: expected at most {}, got {}",
                        i, max_dim, dims[i]
                    ));
                }
            }
        }

        // Check divisibility constraints
        for (dim_idx, divisor) in &constraints.divisible_by {
            if *dim_idx < dims.len() && dims[*dim_idx] % divisor != 0 {
                return VerificationStatus::Error(format!(
                    "Dimension {} not divisible by {}: got {}",
                    dim_idx, divisor, dims[*dim_idx]
                ));
            }
        }

        VerificationStatus::Pass
    }

    /// Check shape patterns
    fn check_pattern(&self, pattern: &ShapePattern, actual: &Shape) -> VerificationStatus {
        match pattern {
            ShapePattern::Matrix => {
                if actual.rank() == 2 {
                    VerificationStatus::Pass
                } else {
                    VerificationStatus::Error(format!(
                        "Expected 2D matrix, got {}D tensor",
                        actual.rank()
                    ))
                }
            }
            ShapePattern::Vector => {
                if actual.rank() == 1 {
                    VerificationStatus::Pass
                } else {
                    VerificationStatus::Error(format!(
                        "Expected 1D vector, got {}D tensor",
                        actual.rank()
                    ))
                }
            }
            ShapePattern::Embedding {
                vocab_size,
                embed_dim,
            } => {
                let dims = actual.dims();
                if dims.len() == 2 && dims[0] == *vocab_size && dims[1] == *embed_dim {
                    VerificationStatus::Pass
                } else {
                    VerificationStatus::Error(format!(
                        "Expected embedding shape [{}, {}], got {:?}",
                        vocab_size, embed_dim, dims
                    ))
                }
            }
        }
    }

    /// Check common patterns for tensors without explicit expectations
    fn check_common_patterns(&self, name: &str, shape: &Shape) -> VerificationStatus {
        // This provides basic sanity checks for common tensor patterns
        let dims = shape.dims();

        if name.contains("embed") {
            if dims.len() != 2 {
                return VerificationStatus::Warning(format!(
                    "Embedding tensor should be 2D, got {}D",
                    dims.len()
                ));
            }
        } else if name.contains("norm") && name.contains("weight") {
            if dims.len() != 1 {
                return VerificationStatus::Warning(format!(
                    "Norm weight should be 1D, got {}D",
                    dims.len()
                ));
            }
        } else if name.contains("bias") {
            if dims.len() != 1 {
                return VerificationStatus::Warning(format!(
                    "Bias tensor should be 1D, got {}D",
                    dims.len()
                ));
            }
        }

        VerificationStatus::Pass
    }

    /// Add CALM model expected shapes
    fn add_calm_expected_shapes(&mut self, config: &CALMConfig) {
        let hidden_size = config.hidden_size as usize;
        let head_dim = hidden_size / config.num_attention_heads as usize;
        let kv_hidden_size = config.num_key_value_heads() as usize * head_dim;

        // Embedding layer
        self.add_exact_shape(
            "transformer.embed_tokens.weight",
            vec![config.vocab_size as usize, hidden_size],
        );

        // Output layer
        self.add_exact_shape(
            "lm_head.weight",
            vec![config.vocab_size as usize, hidden_size],
        );

        // Layer norm
        self.add_exact_shape("transformer.norm.weight", vec![hidden_size]);

        // Add constraints for attention and MLP layers
        let num_layers = config.num_hidden_layers;
        for layer_idx in 0..num_layers {
            let layer_prefix = format!("transformer.layers.{}", layer_idx);

            // Attention projections
            self.add_exact_shape(
                &format!("{}.attention.q_proj.weight", layer_prefix),
                vec![hidden_size, hidden_size],
            );
            self.add_exact_shape(
                &format!("{}.attention.k_proj.weight", layer_prefix),
                vec![kv_hidden_size, hidden_size],
            );
            self.add_exact_shape(
                &format!("{}.attention.v_proj.weight", layer_prefix),
                vec![kv_hidden_size, hidden_size],
            );
            self.add_exact_shape(
                &format!("{}.attention.o_proj.weight", layer_prefix),
                vec![hidden_size, hidden_size],
            );

            // MLP projections
            self.add_exact_shape(
                &format!("{}.mlp.gate_proj.weight", layer_prefix),
                vec![config.intermediate_size as usize, hidden_size],
            );
            self.add_exact_shape(
                &format!("{}.mlp.up_proj.weight", layer_prefix),
                vec![config.intermediate_size as usize, hidden_size],
            );
            self.add_exact_shape(
                &format!("{}.mlp.down_proj.weight", layer_prefix),
                vec![hidden_size, config.intermediate_size as usize],
            );

            // Layer norms
            self.add_exact_shape(
                &format!("{}.input_layernorm.weight", layer_prefix),
                vec![hidden_size],
            );
            self.add_exact_shape(
                &format!("{}.post_attention_layernorm.weight", layer_prefix),
                vec![hidden_size],
            );
        }
    }

    /// Add autoencoder expected shapes
    fn add_autoencoder_expected_shapes(&mut self, config: &AutoencoderConfig) {
        // Add autoencoder-specific shape expectations
        // Using tensor names that match the autoencoder loader (encoder.* and decoder.*)
        self.add_exact_shape(
            "encoder.embed_tokens.weight",
            vec![config.vocab_size as usize, config.hidden_size as usize],
        );

        self.add_exact_shape(
            "decoder.embed_tokens.weight",
            vec![config.vocab_size as usize, config.hidden_size as usize],
        );

        // Add encoder layer norms
        for layer_idx in 0..config.num_encoder_layers {
            self.add_exact_shape(
                &format!("encoder.layers.{}.norm.weight", layer_idx),
                vec![config.hidden_size as usize],
            );
        }

        // Add decoder layer norms
        for layer_idx in 0..config.num_decoder_layers {
            self.add_exact_shape(
                &format!("decoder.layers.{}.norm.weight", layer_idx),
                vec![config.hidden_size as usize],
            );
        }
    }
}

/// Expected shape specification
#[derive(Debug, Clone)]
pub enum ExpectedShape {
    /// Exact shape match required
    Exact(Vec<usize>),
    /// Shape must satisfy constraints
    Constrained(ShapeConstraints),
    /// Shape must match pattern
    Pattern(ShapePattern),
}

/// Shape constraints for flexible matching
#[derive(Debug, Clone)]
pub struct ShapeConstraints {
    /// Expected rank (number of dimensions)
    pub rank: Option<usize>,
    /// Minimum values for each dimension
    pub min_dims: Option<Vec<usize>>,
    /// Maximum values for each dimension
    pub max_dims: Option<Vec<usize>>,
    /// Dimensions that must be divisible by certain values
    pub divisible_by: HashMap<usize, usize>,
}

impl ShapeConstraints {
    /// Create constraints with only rank requirement
    pub fn rank_only(rank: usize) -> Self {
        Self {
            rank: Some(rank),
            min_dims: None,
            max_dims: None,
            divisible_by: HashMap::new(),
        }
    }

    /// Add divisibility constraint
    pub fn add_divisibility_constraint(mut self, dim_idx: usize, divisor: usize) -> Self {
        self.divisible_by.insert(dim_idx, divisor);
        self
    }
}

/// Common shape patterns
#[derive(Debug, Clone)]
pub enum ShapePattern {
    /// 2D matrix
    Matrix,
    /// 1D vector
    Vector,
    /// Embedding matrix with specific vocab size and embedding dimension
    Embedding { vocab_size: usize, embed_dim: usize },
}

/// Verification status for a tensor
#[derive(Debug, Clone)]
pub enum VerificationStatus {
    /// Shape verification passed
    Pass,
    /// Warning (non-critical issue)
    Warning(String),
    /// Error (critical issue)
    Error(String),
}

/// Result of verifying a single tensor
#[derive(Debug, Clone)]
pub struct TensorVerificationResult {
    pub tensor_name: String,
    pub expected_shape: Option<ExpectedShape>,
    pub actual_shape: Option<Shape>,
    pub status: VerificationStatus,
}

/// Complete verification report
#[derive(Debug)]
pub struct VerificationReport {
    pub total_tensors: usize,
    pub results: Vec<TensorVerificationResult>,
    pub error_count: usize,
    pub warning_count: usize,
    pub strict_mode: bool,
}

impl VerificationReport {
    /// Check if verification passed (no errors)
    pub fn passed(&self) -> bool {
        self.error_count == 0
    }

    /// Get all errors
    pub fn errors(&self) -> Vec<&TensorVerificationResult> {
        self.results
            .iter()
            .filter(|r| matches!(r.status, VerificationStatus::Error(_)))
            .collect()
    }

    /// Get all warnings
    pub fn warnings(&self) -> Vec<&TensorVerificationResult> {
        self.results
            .iter()
            .filter(|r| matches!(r.status, VerificationStatus::Warning(_)))
            .collect()
    }

    /// Display summary
    pub fn display_summary(&self) {
        info!("Shape Verification Report:");
        info!("  Total tensors: {}", self.total_tensors);
        info!("  Errors: {}", self.error_count);
        info!("  Warnings: {}", self.warning_count);
        info!("  Strict mode: {}", self.strict_mode);
        info!(
            "  Overall status: {}",
            if self.passed() { "PASS" } else { "FAIL" }
        );

        if self.error_count > 0 {
            error!("Errors found:");
            for error_result in self.errors() {
                if let VerificationStatus::Error(msg) = &error_result.status {
                    error!("  {}: {}", error_result.tensor_name, msg);
                }
            }
        }

        if self.warning_count > 0 {
            warn!("Warnings found:");
            for warning_result in self.warnings() {
                if let VerificationStatus::Warning(msg) = &warning_result.status {
                    warn!("  {}: {}", warning_result.tensor_name, msg);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    fn create_test_tensor(shape: &[usize]) -> Tensor {
        Tensor::zeros(shape, DType::F32, &Device::Cpu).unwrap()
    }

    #[test]
    fn test_exact_shape_verification() {
        let mut verifier = ShapeVerifier::new(true);
        verifier.add_exact_shape("test_tensor", vec![100, 200]);

        let tensor = create_test_tensor(&[100, 200]);
        let result = verifier.verify_tensor_shape("test_tensor", &tensor);

        assert!(matches!(result.status, VerificationStatus::Pass));
    }

    #[test]
    fn test_shape_mismatch() {
        let mut verifier = ShapeVerifier::new(true);
        verifier.add_exact_shape("test_tensor", vec![100, 200]);

        let tensor = create_test_tensor(&[150, 200]);
        let result = verifier.verify_tensor_shape("test_tensor", &tensor);

        assert!(matches!(result.status, VerificationStatus::Error(_)));
    }

    #[test]
    fn test_shape_constraints() {
        let mut verifier = ShapeVerifier::new(false);
        let constraints = ShapeConstraints::rank_only(2).add_divisibility_constraint(0, 8); // First dimension must be divisible by 8

        verifier.add_constrained_shape("test_tensor", constraints);

        // Should pass
        let tensor = create_test_tensor(&[96, 200]); // 96 is divisible by 8
        let result = verifier.verify_tensor_shape("test_tensor", &tensor);
        assert!(matches!(result.status, VerificationStatus::Pass));

        // Should fail
        let tensor = create_test_tensor(&[97, 200]); // 97 is not divisible by 8
        let result = verifier.verify_tensor_shape("test_tensor", &tensor);
        assert!(matches!(result.status, VerificationStatus::Error(_)));
    }

    #[test]
    fn test_pattern_verification() {
        let mut verifier = ShapeVerifier::new(true);
        verifier.add_expected_shape(
            "embedding",
            ExpectedShape::Pattern(ShapePattern::Embedding {
                vocab_size: 1000,
                embed_dim: 512,
            }),
        );

        let tensor = create_test_tensor(&[1000, 512]);
        let result = verifier.verify_tensor_shape("embedding", &tensor);
        assert!(matches!(result.status, VerificationStatus::Pass));
    }

    #[test]
    fn test_calm_config_shapes() {
        let config = CALMConfig::default();
        let verifier = ShapeVerifier::for_calm_model(&config, false);

        assert!(verifier.expected_shapes.len() > 0);
        assert!(verifier
            .expected_shapes
            .contains_key("transformer.embed_tokens.weight"));
        assert!(verifier.expected_shapes.contains_key("lm_head.weight"));
    }

    #[test]
    fn test_calm_config_gqa_kv_shapes() {
        let mut config = CALMConfig::default();
        config.hidden_size = 1024;
        config.num_attention_heads = 16;
        config.num_key_value_heads = Some(4);

        let verifier = ShapeVerifier::for_calm_model(&config, false);
        let expected_kv_shape = vec![256, 1024];

        match verifier
            .expected_shapes
            .get("transformer.layers.0.attention.k_proj.weight")
        {
            Some(ExpectedShape::Exact(dims)) => assert_eq!(dims, &expected_kv_shape),
            other => panic!("unexpected k_proj expected shape: {:?}", other),
        }

        match verifier
            .expected_shapes
            .get("transformer.layers.0.attention.v_proj.weight")
        {
            Some(ExpectedShape::Exact(dims)) => assert_eq!(dims, &expected_kv_shape),
            other => panic!("unexpected v_proj expected shape: {:?}", other),
        }
    }

    #[test]
    fn test_model_verification() {
        let mut tensors = HashMap::new();
        tensors.insert("test_weight".to_string(), create_test_tensor(&[100, 200]));

        let verifier = ShapeVerifier::new(false);
        let report = verifier.verify_model_shapes(&tensors).unwrap();

        assert_eq!(report.total_tensors, 1);
        assert!(report.passed());
        report.display_summary();
    }

    #[test]
    fn test_common_patterns() {
        let verifier = ShapeVerifier::new(false);

        // Test embedding pattern
        let embed_tensor = create_test_tensor(&[1000, 512]);
        let result = verifier.verify_tensor_shape("model.embed_tokens.weight", &embed_tensor);
        assert!(matches!(result.status, VerificationStatus::Pass));

        // Test norm pattern
        let norm_tensor = create_test_tensor(&[512]);
        let result = verifier.verify_tensor_shape("model.norm.weight", &norm_tensor);
        assert!(matches!(result.status, VerificationStatus::Pass));

        // Test bias pattern warning
        let bias_tensor = create_test_tensor(&[512, 10]); // Should be 1D
        let result = verifier.verify_tensor_shape("model.bias", &bias_tensor);
        assert!(matches!(result.status, VerificationStatus::Warning(_)));
    }
}
