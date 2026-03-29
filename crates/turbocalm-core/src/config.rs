use serde::{Deserialize, Serialize};

/// Configuration for CALM (Continuous Autoregressive Language Model)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct CALMConfig {
    // CALM-specific parameters
    pub ae_path: Option<String>,
    pub model_type: String,
    pub patch_size: u32,
    pub num_mlp_layers: u32,
    pub num_samples: u32,
    pub beta: f64,
    pub noise_size: u32,
    pub latent_size: u32,

    // Standard transformer parameters
    pub vocab_size: u32,
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: Option<u32>,
    pub hidden_act: String,
    pub max_position_embeddings: u32,
    pub initializer_range: f64,
    pub rms_norm_eps: f64,
    pub use_cache: bool,
    pub pretraining_tp: u32,
    pub tie_word_embeddings: bool,

    // RoPE parameters
    pub rope_theta: f64,
    pub rope_scaling: Option<RopeScaling>,

    // Attention parameters
    pub attention_bias: bool,
    pub attention_dropout: f64,

    // MLP parameters
    pub mlp_bias: bool,

    // Token IDs
    pub pad_token_id: Option<u32>,
    pub bos_token_id: u32,
    pub eos_token_id: u32,

    // HuggingFace-specific fields (optional, for compatibility)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub architectures: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub torch_dtype: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transformers_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temp: Option<f64>,
}

impl Default for CALMConfig {
    fn default() -> Self {
        Self {
            ae_path: None,
            model_type: "energy".to_string(),
            patch_size: 4,
            num_mlp_layers: 4,
            num_samples: 8,
            beta: 1.0,
            noise_size: 64,
            latent_size: 128,
            vocab_size: 32000,
            hidden_size: 768,
            intermediate_size: 2048,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            num_key_value_heads: Some(12),
            hidden_act: "silu".to_string(),
            max_position_embeddings: 2048,
            initializer_range: 0.02,
            rms_norm_eps: 1e-6,
            use_cache: true,
            pretraining_tp: 1,
            tie_word_embeddings: false,
            rope_theta: 10000.0,
            rope_scaling: None,
            attention_bias: false,
            attention_dropout: 0.0,
            mlp_bias: false,
            pad_token_id: None,
            bos_token_id: 1,
            eos_token_id: 2,
            architectures: None,
            torch_dtype: None,
            transformers_version: None,
            temp: None,
        }
    }
}

/// Configuration for the Autoencoder component
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AutoencoderConfig {
    // Autoencoder-specific parameters
    pub ae_dropout: f64,
    pub kl_clamp: f64,
    pub kl_weight: f64,
    pub patch_size: u32,
    pub latent_size: u32,

    // Transformer parameters
    pub vocab_size: u32,
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub num_encoder_layers: u32,
    pub num_decoder_layers: u32,
    pub hidden_act: String,
    pub max_position_embeddings: u32,
    pub initializer_range: f64,
    pub rms_norm_eps: f64,
    pub pretraining_tp: u32,
    pub mlp_bias: bool,

    // Token IDs
    pub pad_token_id: Option<u32>,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub tie_word_embeddings: bool,
}

impl Default for AutoencoderConfig {
    fn default() -> Self {
        Self {
            ae_dropout: 0.15,
            kl_clamp: 0.5,
            kl_weight: 1e-3,
            patch_size: 4,
            latent_size: 128,
            vocab_size: 32000,
            hidden_size: 512,
            intermediate_size: 1280,
            num_encoder_layers: 2,
            num_decoder_layers: 2,
            hidden_act: "silu".to_string(),
            max_position_embeddings: 2048,
            initializer_range: 0.02,
            rms_norm_eps: 1e-6,
            pretraining_tp: 1,
            mlp_bias: false,
            pad_token_id: None,
            bos_token_id: 1,
            eos_token_id: 2,
            tie_word_embeddings: false,
        }
    }
}

/// RoPE scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RopeScaling {
    pub rope_type: String,
    pub factor: f64,
}

impl RopeScaling {
    /// Validates the RoPE scaling configuration
    pub fn validate(&self) -> Result<(), String> {
        if !["linear", "dynamic"].contains(&self.rope_type.as_str()) {
            return Err(format!(
                "rope_scaling type must be one of ['linear', 'dynamic'], got {}",
                self.rope_type
            ));
        }
        if self.factor <= 1.0 {
            return Err(format!(
                "rope_scaling factor must be > 1.0, got {}",
                self.factor
            ));
        }
        Ok(())
    }
}

impl CALMConfig {
    /// Load configuration from JSON file
    pub fn from_json_file(path: &str) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&content)?;
        config.validate()?;
        Ok(config)
    }

    /// Save configuration to JSON file
    pub fn to_json_file(&self, path: &str) -> anyhow::Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Validate the configuration
    pub fn validate(&self) -> anyhow::Result<()> {
        if let Some(ref rope_scaling) = self.rope_scaling {
            rope_scaling.validate().map_err(|e| anyhow::anyhow!(e))?;
        }

        if self.num_attention_heads == 0 {
            return Err(anyhow::anyhow!("num_attention_heads must be > 0"));
        }

        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(anyhow::anyhow!(
                "hidden_size must be divisible by num_attention_heads"
            ));
        }

        Ok(())
    }

    /// Get the actual number of key-value heads (defaults to attention heads if not set)
    pub fn num_key_value_heads(&self) -> u32 {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }
}

impl AutoencoderConfig {
    /// Load configuration from JSON file
    pub fn from_json_file(path: &str) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&content)?;
        Ok(config)
    }

    /// Save configuration to JSON file
    pub fn to_json_file(&self, path: &str) -> anyhow::Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calm_config_default() {
        let config = CALMConfig::default();
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.patch_size, 4);
        assert_eq!(config.model_type, "energy");
    }

    #[test]
    fn test_autoencoder_config_default() {
        let config = AutoencoderConfig::default();
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.hidden_size, 512);
        assert_eq!(config.ae_dropout, 0.15);
        assert_eq!(config.patch_size, 4);
    }

    #[test]
    fn test_rope_scaling_validation() {
        let valid_rope = RopeScaling {
            rope_type: "linear".to_string(),
            factor: 2.0,
        };
        assert!(valid_rope.validate().is_ok());

        let invalid_type = RopeScaling {
            rope_type: "invalid".to_string(),
            factor: 2.0,
        };
        assert!(invalid_type.validate().is_err());

        let invalid_factor = RopeScaling {
            rope_type: "linear".to_string(),
            factor: 0.5,
        };
        assert!(invalid_factor.validate().is_err());
    }

    #[test]
    fn test_calm_config_validation() {
        let mut config = CALMConfig::default();
        assert!(config.validate().is_ok());

        // Test invalid attention heads
        config.num_attention_heads = 0;
        assert!(config.validate().is_err());

        // Test misaligned hidden size
        config.num_attention_heads = 13; // 768 not divisible by 13
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_json_serialization() {
        let config = CALMConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let parsed: CALMConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config, parsed);
    }

    #[test]
    fn test_hf_config_compatibility() {
        // Test parsing a HuggingFace-style config with extra fields
        let hf_config_json = r#"{
            "ae_path": "/some/path",
            "architectures": ["EnergyTransformer"],
            "attention_bias": false,
            "attention_dropout": 0.0,
            "beta": 1.0,
            "bos_token_id": 128000,
            "eos_token_id": 128001,
            "hidden_act": "silu",
            "hidden_size": 1024,
            "initializer_range": 0.02,
            "intermediate_size": 2752,
            "latent_size": 128,
            "max_position_embeddings": 2048,
            "mlp_bias": false,
            "noise_size": 64,
            "num_attention_heads": 16,
            "num_hidden_layers": 16,
            "num_key_value_heads": 16,
            "num_mlp_layers": 4,
            "pad_token_id": 128256,
            "patch_size": 4,
            "pretraining_tp": 1,
            "rms_norm_eps": 1e-06,
            "rope_scaling": null,
            "rope_theta": 10000.0,
            "temp": 1.0,
            "tie_word_embeddings": false,
            "torch_dtype": "float32",
            "transformers_version": "4.43.0",
            "use_cache": true,
            "vocab_size": 128257,
            "model_type": "energy"
        }"#;

        let config: CALMConfig = serde_json::from_str(hf_config_json).unwrap();

        // Verify critical values match the real CALM-M config
        assert_eq!(config.vocab_size, 128257);
        assert_eq!(config.hidden_size, 1024);
        assert_eq!(config.bos_token_id, 128000);
        assert_eq!(config.eos_token_id, 128001);
        assert_eq!(config.pad_token_id, Some(128256));
        assert_eq!(config.num_hidden_layers, 16);
        assert_eq!(config.num_attention_heads, 16);
        assert_eq!(config.intermediate_size, 2752);

        // Verify HF-specific fields are captured
        assert_eq!(
            config.architectures,
            Some(vec!["EnergyTransformer".to_string()])
        );
        assert_eq!(config.torch_dtype, Some("float32".to_string()));
        assert_eq!(config.transformers_version, Some("4.43.0".to_string()));
        assert_eq!(config.temp, Some(1.0));
    }
}
