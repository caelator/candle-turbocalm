use anyhow::Result;
use std::collections::HashMap;
use candle_core::Tensor;
use tracing::{debug, info, warn};

/// Tensor name remapper for converting between different model naming conventions
pub struct TensorNameRemapper {
    mapping_rules: Vec<MappingRule>,
}

impl TensorNameRemapper {
    /// Create a new tensor name remapper
    pub fn new() -> Self {
        Self {
            mapping_rules: Vec::new(),
        }
    }

    /// Create a remapper with CALM-specific rules
    pub fn for_calm() -> Self {
        let mut remapper = Self::new();
        remapper.add_calm_mapping_rules();
        remapper
    }

    /// Create a remapper for converting from HuggingFace Llama to CALM
    pub fn llama_to_calm() -> Self {
        let mut remapper = Self::new();
        remapper.add_llama_to_calm_rules();
        remapper
    }

    /// Add a simple string replacement rule
    pub fn add_replacement_rule(&mut self, from: &str, to: &str) {
        self.mapping_rules.push(MappingRule::StringReplace {
            from: from.to_string(),
            to: to.to_string(),
        });
    }

    /// Add a regex-based replacement rule
    pub fn add_regex_rule(&mut self, pattern: &str, replacement: &str) -> Result<()> {
        let regex = regex::Regex::new(pattern)?;
        self.mapping_rules.push(MappingRule::RegexReplace {
            pattern: regex,
            replacement: replacement.to_string(),
        });
        Ok(())
    }

    /// Add a custom mapping function
    pub fn add_custom_rule<F>(&mut self, func: F)
    where
        F: Fn(&str) -> Option<String> + Send + Sync + 'static,
    {
        self.mapping_rules.push(MappingRule::Custom {
            func: Box::new(func),
        });
    }

    /// Apply remapping to a tensor dictionary
    pub fn remap_tensors(
        &self,
        tensors: HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>> {
        info!("Remapping {} tensors", tensors.len());

        let mut remapped = HashMap::new();
        let mut unmapped_count = 0;

        for (original_name, tensor) in tensors {
            if let Some(new_name) = self.map_tensor_name(&original_name) {
                debug!("Remapped: {} -> {}", original_name, new_name);
                remapped.insert(new_name, tensor);
            } else {
                debug!("No mapping for: {}", original_name);
                remapped.insert(original_name, tensor);
                unmapped_count += 1;
            }
        }

        if unmapped_count > 0 {
            warn!("{} tensors had no mapping rules applied", unmapped_count);
        }

        info!("Remapping complete: {} tensors", remapped.len());
        Ok(remapped)
    }

    /// Map a single tensor name
    pub fn map_tensor_name(&self, name: &str) -> Option<String> {
        let mut current_name = name.to_string();

        for rule in &self.mapping_rules {
            if let Some(mapped) = rule.apply(&current_name) {
                current_name = mapped;
            }
        }

        if current_name != name {
            Some(current_name)
        } else {
            None
        }
    }

    /// Add standard CALM mapping rules
    fn add_calm_mapping_rules(&mut self) {
        // Standard transformer layer mappings
        self.add_replacement_rule("model.embed_tokens.weight", "transformer.embed_tokens.weight");
        self.add_replacement_rule("model.norm.weight", "transformer.norm.weight");
        self.add_replacement_rule("lm_head.weight", "lm_head.weight");

        // Attention layer mappings
        self.add_replacement_rule(".self_attn.q_proj.weight", ".attention.q_proj.weight");
        self.add_replacement_rule(".self_attn.k_proj.weight", ".attention.k_proj.weight");
        self.add_replacement_rule(".self_attn.v_proj.weight", ".attention.v_proj.weight");
        self.add_replacement_rule(".self_attn.o_proj.weight", ".attention.o_proj.weight");

        // MLP mappings
        self.add_replacement_rule(".mlp.gate_proj.weight", ".mlp.gate_proj.weight");
        self.add_replacement_rule(".mlp.up_proj.weight", ".mlp.up_proj.weight");
        self.add_replacement_rule(".mlp.down_proj.weight", ".mlp.down_proj.weight");

        // Layer norm mappings
        self.add_replacement_rule(".input_layernorm.weight", ".input_layernorm.weight");
        self.add_replacement_rule(".post_attention_layernorm.weight", ".post_attention_layernorm.weight");

        // CALM-specific mappings
        self.add_replacement_rule("model.layers.", "transformer.layers.");

        // Generative head mappings (CALM-specific)
        self.add_replacement_rule("generative_head.", "generative_head.");
        self.add_replacement_rule("ae_model.", "ae_model.");
    }

    /// Add Llama to CALM conversion rules
    fn add_llama_to_calm_rules(&mut self) {
        info!("Adding Llama to CALM conversion rules");

        // Base model structure
        self.add_replacement_rule("model.embed_tokens", "transformer.embed_tokens");
        self.add_replacement_rule("model.norm", "transformer.norm");
        self.add_replacement_rule("model.layers", "transformer.layers");

        // Keep lm_head as is for compatibility
        // self.add_replacement_rule("lm_head", "lm_head");

        // Add CALM-specific components with default initialization markers
        // These would be handled specially during loading
    }

    /// Get a summary of mapping rules
    pub fn get_mapping_summary(&self) -> MappingSummary {
        let mut rule_types = HashMap::new();
        for rule in &self.mapping_rules {
            let rule_type = match rule {
                MappingRule::StringReplace { .. } => "StringReplace",
                MappingRule::RegexReplace { .. } => "RegexReplace",
                MappingRule::Custom { .. } => "Custom",
            };
            *rule_types.entry(rule_type.to_string()).or_insert(0) += 1;
        }

        MappingSummary {
            total_rules: self.mapping_rules.len(),
            rule_types,
        }
    }

    /// Create a reverse mapping (useful for debugging)
    pub fn create_reverse_mapping(&self, tensor_names: &[String]) -> HashMap<String, String> {
        let mut reverse_map = HashMap::new();

        for name in tensor_names {
            if let Some(mapped_name) = self.map_tensor_name(name) {
                reverse_map.insert(mapped_name, name.clone());
            }
        }

        reverse_map
    }
}

impl Default for TensorNameRemapper {
    fn default() -> Self {
        Self::new()
    }
}

/// Individual mapping rule
#[allow(dead_code)]
enum MappingRule {
    StringReplace {
        from: String,
        to: String,
    },
    RegexReplace {
        pattern: regex::Regex,
        replacement: String,
    },
    Custom {
        func: Box<dyn Fn(&str) -> Option<String> + Send + Sync>,
    },
}

impl MappingRule {
    /// Apply this mapping rule to a tensor name
    fn apply(&self, name: &str) -> Option<String> {
        match self {
            MappingRule::StringReplace { from, to } => {
                if name.contains(from) {
                    Some(name.replace(from, to))
                } else {
                    None
                }
            }
            MappingRule::RegexReplace { pattern, replacement } => {
                if pattern.is_match(name) {
                    Some(pattern.replace_all(name, replacement).into_owned())
                } else {
                    None
                }
            }
            MappingRule::Custom { func } => func(name),
        }
    }
}

/// Summary of mapping rules
#[derive(Debug, Clone)]
pub struct MappingSummary {
    pub total_rules: usize,
    pub rule_types: HashMap<String, usize>,
}

impl MappingSummary {
    /// Display mapping summary
    pub fn display_summary(&self) {
        info!("Mapping Rules Summary:");
        info!("  Total rules: {}", self.total_rules);
        for (rule_type, count) in &self.rule_types {
            info!("  {}: {} rules", rule_type, count);
        }
    }
}

/// Predefined remapping configurations
pub struct RemappingPresets;

impl RemappingPresets {
    /// Get HuggingFace Llama to CALM remapping
    pub fn huggingface_llama_to_calm() -> TensorNameRemapper {
        let mut remapper = TensorNameRemapper::new();

        // Embedding and output layers
        remapper.add_replacement_rule("model.embed_tokens.weight", "transformer.embed_tokens.weight");
        remapper.add_replacement_rule("model.norm.weight", "transformer.norm.weight");
        remapper.add_replacement_rule("lm_head.weight", "lm_head.weight");

        // Transformer layers
        remapper.add_replacement_rule("model.layers.", "transformer.layers.");

        // Attention components
        remapper.add_replacement_rule(".self_attn.q_proj.weight", ".attention.q_proj.weight");
        remapper.add_replacement_rule(".self_attn.k_proj.weight", ".attention.k_proj.weight");
        remapper.add_replacement_rule(".self_attn.v_proj.weight", ".attention.v_proj.weight");
        remapper.add_replacement_rule(".self_attn.o_proj.weight", ".attention.o_proj.weight");

        // Optional bias terms
        remapper.add_replacement_rule(".self_attn.q_proj.bias", ".attention.q_proj.bias");
        remapper.add_replacement_rule(".self_attn.k_proj.bias", ".attention.k_proj.bias");
        remapper.add_replacement_rule(".self_attn.v_proj.bias", ".attention.v_proj.bias");
        remapper.add_replacement_rule(".self_attn.o_proj.bias", ".attention.o_proj.bias");

        remapper
    }

    /// Get CALM to HuggingFace Llama remapping (reverse)
    pub fn calm_to_huggingface_llama() -> TensorNameRemapper {
        let mut remapper = TensorNameRemapper::new();

        // Reverse mappings
        remapper.add_replacement_rule("transformer.embed_tokens.weight", "model.embed_tokens.weight");
        remapper.add_replacement_rule("transformer.norm.weight", "model.norm.weight");
        remapper.add_replacement_rule("transformer.layers.", "model.layers.");

        // Reverse attention mappings
        remapper.add_replacement_rule(".attention.q_proj.weight", ".self_attn.q_proj.weight");
        remapper.add_replacement_rule(".attention.k_proj.weight", ".self_attn.k_proj.weight");
        remapper.add_replacement_rule(".attention.v_proj.weight", ".self_attn.v_proj.weight");
        remapper.add_replacement_rule(".attention.o_proj.weight", ".self_attn.o_proj.weight");

        remapper
    }

    /// Get identity mapping (no changes)
    pub fn identity() -> TensorNameRemapper {
        TensorNameRemapper::new()
    }
}

/// Utilities for working with tensor remapping
pub struct RemappingUtils;

impl RemappingUtils {
    /// Analyze tensor name patterns in a model
    pub fn analyze_tensor_patterns(tensor_names: &[String]) -> TensorPatternAnalysis {
        let mut layer_counts = HashMap::new();
        let mut component_types = HashMap::new();
        let mut prefixes = HashMap::new();

        for name in tensor_names {
            // Count layers
            if let Some(layer_num) = extract_layer_number(name) {
                *layer_counts.entry(layer_num).or_insert(0) += 1;
            }

            // Count component types
            let component = identify_component_type(name);
            *component_types.entry(component).or_insert(0) += 1;

            // Count prefixes
            let prefix = name.split('.').next().unwrap_or("").to_string();
            *prefixes.entry(prefix).or_insert(0) += 1;
        }

        TensorPatternAnalysis {
            total_tensors: tensor_names.len(),
            layer_counts,
            component_types,
            prefixes,
        }
    }

    /// Validate that a remapping preserves expected tensor structure
    pub fn validate_remapping(
        original_names: &[String],
        remapped_names: &[String],
    ) -> RemappingValidation {
        let original_analysis = Self::analyze_tensor_patterns(original_names);
        let remapped_analysis = Self::analyze_tensor_patterns(remapped_names);

        let tensor_count_preserved = original_analysis.total_tensors == remapped_analysis.total_tensors;

        RemappingValidation {
            tensor_count_preserved,
            original_analysis,
            remapped_analysis,
        }
    }
}

/// Extract layer number from tensor name
fn extract_layer_number(name: &str) -> Option<u32> {
    for part in name.split('.') {
        if let Ok(num) = part.parse::<u32>() {
            return Some(num);
        }
    }
    None
}

/// Identify component type from tensor name
fn identify_component_type(name: &str) -> String {
    if name.contains("embed") {
        "embedding".to_string()
    } else if name.contains("norm") {
        "normalization".to_string()
    } else if name.contains("attn") || name.contains("attention") {
        "attention".to_string()
    } else if name.contains("mlp") {
        "mlp".to_string()
    } else if name.contains("lm_head") {
        "output".to_string()
    } else if name.contains("generative_head") {
        "generative".to_string()
    } else if name.contains("ae_model") {
        "autoencoder".to_string()
    } else {
        "other".to_string()
    }
}

/// Analysis of tensor name patterns
#[derive(Debug, Clone)]
pub struct TensorPatternAnalysis {
    pub total_tensors: usize,
    pub layer_counts: HashMap<u32, usize>,
    pub component_types: HashMap<String, usize>,
    pub prefixes: HashMap<String, usize>,
}

impl TensorPatternAnalysis {
    /// Display pattern analysis
    pub fn display_analysis(&self) {
        info!("Tensor Pattern Analysis:");
        info!("  Total tensors: {}", self.total_tensors);
        info!("  Layer distribution: {:?}", self.layer_counts);
        info!("  Component types: {:?}", self.component_types);
        info!("  Top prefixes: {:?}", self.prefixes);
    }
}

/// Validation result for tensor remapping

pub struct RemappingValidation {
    pub tensor_count_preserved: bool,
    pub original_analysis: TensorPatternAnalysis,
    pub remapped_analysis: TensorPatternAnalysis,
}

impl RemappingValidation {
    /// Check if remapping passed validation
    pub fn is_valid(&self) -> bool {
        self.tensor_count_preserved
    }

    /// Display validation results
    pub fn display_validation(&self) {
        info!("Remapping Validation:");
        info!("  Tensor count preserved: {}", self.tensor_count_preserved);
        info!("  Original pattern:");
        self.original_analysis.display_analysis();
        info!("  Remapped pattern:");
        self.remapped_analysis.display_analysis();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_remapping() {
        let mut remapper = TensorNameRemapper::new();
        remapper.add_replacement_rule("model.embed_tokens", "transformer.embed_tokens");

        let original = "model.embed_tokens.weight";
        let mapped = remapper.map_tensor_name(original);
        assert_eq!(mapped, Some("transformer.embed_tokens.weight".to_string()));
    }

    #[test]
    fn test_calm_remapper() {
        let remapper = TensorNameRemapper::for_calm();
        let summary = remapper.get_mapping_summary();
        assert!(summary.total_rules > 0);
        summary.display_summary();
    }

    #[test]
    fn test_llama_to_calm_preset() {
        let remapper = RemappingPresets::huggingface_llama_to_calm();

        let original = "model.layers.0.self_attn.q_proj.weight";
        let mapped = remapper.map_tensor_name(original);
        assert_eq!(mapped, Some("transformer.layers.0.attention.q_proj.weight".to_string()));
    }

    #[test]
    fn test_tensor_pattern_analysis() {
        let tensor_names = vec![
            "model.embed_tokens.weight".to_string(),
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            "model.layers.1.self_attn.q_proj.weight".to_string(),
            "model.norm.weight".to_string(),
        ];

        let analysis = RemappingUtils::analyze_tensor_patterns(&tensor_names);
        assert_eq!(analysis.total_tensors, 4);
        analysis.display_analysis();
    }

    #[test]
    fn test_component_identification() {
        assert_eq!(identify_component_type("model.embed_tokens.weight"), "embedding");
        assert_eq!(identify_component_type("layers.0.self_attn.q_proj.weight"), "attention");
        assert_eq!(identify_component_type("layers.0.mlp.gate_proj.weight"), "mlp");
        assert_eq!(identify_component_type("model.norm.weight"), "normalization");
        assert_eq!(identify_component_type("lm_head.weight"), "output");
    }

    #[test]
    fn test_layer_number_extraction() {
        assert_eq!(extract_layer_number("model.layers.0.self_attn.q_proj.weight"), Some(0));
        assert_eq!(extract_layer_number("model.layers.15.mlp.gate_proj.weight"), Some(15));
        assert_eq!(extract_layer_number("model.embed_tokens.weight"), None);
    }

    #[test]
    fn test_remapping_validation() {
        let original_names = vec![
            "model.embed_tokens.weight".to_string(),
            "model.layers.0.self_attn.q_proj.weight".to_string(),
        ];

        let remapped_names = vec![
            "transformer.embed_tokens.weight".to_string(),
            "transformer.layers.0.attention.q_proj.weight".to_string(),
        ];

        let validation = RemappingUtils::validate_remapping(&original_names, &remapped_names);
        assert!(validation.is_valid());
        validation.display_validation();
    }
}