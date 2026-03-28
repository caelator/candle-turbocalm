use std::fmt;

/// Main error type for turbocalm-core operations
#[derive(Debug)]
pub enum TurboCALMError {
    /// Configuration-related errors
    Config(ConfigError),
    /// Device-related errors
    Device(DeviceError),
    /// Model loading and initialization errors
    Model(ModelError),
    /// Tokenizer-related errors
    Tokenizer(TokenizerError),
    /// HuggingFace Hub errors
    Hub(HubError),
    /// Tensor computation errors
    Tensor(TensorError),
    /// I/O errors
    Io(std::io::Error),
    /// Candle framework errors
    Candle(candle_core::Error),
    /// JSON serialization/deserialization errors
    Json(serde_json::Error),
    /// Generic errors with custom message
    Generic(String),
}

/// Configuration-related errors
#[derive(Debug)]
pub enum ConfigError {
    /// Invalid configuration parameter
    InvalidParameter { param: String, reason: String },
    /// Missing required configuration
    MissingRequired(String),
    /// Configuration file not found
    FileNotFound(String),
    /// Invalid JSON format
    InvalidFormat(String),
    /// Validation failed
    ValidationFailed(String),
}

/// Device-related errors
#[derive(Debug)]
pub enum DeviceError {
    /// No suitable device found
    NoDeviceAvailable,
    /// Specific device type not available
    DeviceTypeUnavailable(String),
    /// Device initialization failed
    InitializationFailed(String),
    /// Device feature not enabled
    FeatureNotEnabled(String),
}

/// Model loading and initialization errors
#[derive(Debug)]
pub enum ModelError {
    /// Model file not found
    FileNotFound(String),
    /// Incompatible model version
    IncompatibleVersion { expected: String, found: String },
    /// Missing model component
    MissingComponent(String),
    /// Model loading failed
    LoadingFailed(String),
    /// Model architecture mismatch
    ArchitectureMismatch(String),
}

/// Tokenizer-related errors
#[derive(Debug)]
pub enum TokenizerError {
    /// Tokenizer loading failed
    LoadingFailed(String),
    /// Tokenization failed
    TokenizationFailed(String),
    /// Unsupported tokenizer type
    UnsupportedType(String),
    /// Vocab file issues
    VocabError(String),
}

/// HuggingFace Hub errors
#[derive(Debug)]
pub enum HubError {
    /// Authentication failed
    AuthenticationFailed,
    /// Model not found on hub
    ModelNotFound(String),
    /// Download failed
    DownloadFailed(String),
    /// Network error
    NetworkError(String),
    /// Invalid model repository
    InvalidRepository(String),
}

/// Tensor computation errors
#[derive(Debug)]
pub enum TensorError {
    /// Shape mismatch
    ShapeMismatch { expected: String, found: String },
    /// Unsupported operation
    UnsupportedOperation(String),
    /// Computation failed
    ComputationFailed(String),
    /// Memory allocation failed
    OutOfMemory,
    /// Type conversion failed
    TypeConversionFailed(String),
}

impl fmt::Display for TurboCALMError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TurboCALMError::Config(e) => write!(f, "Configuration error: {}", e),
            TurboCALMError::Device(e) => write!(f, "Device error: {}", e),
            TurboCALMError::Model(e) => write!(f, "Model error: {}", e),
            TurboCALMError::Tokenizer(e) => write!(f, "Tokenizer error: {}", e),
            TurboCALMError::Hub(e) => write!(f, "HuggingFace Hub error: {}", e),
            TurboCALMError::Tensor(e) => write!(f, "Tensor error: {}", e),
            TurboCALMError::Io(e) => write!(f, "I/O error: {}", e),
            TurboCALMError::Candle(e) => write!(f, "Candle error: {}", e),
            TurboCALMError::Json(e) => write!(f, "JSON error: {}", e),
            TurboCALMError::Generic(msg) => write!(f, "{}", msg),
        }
    }
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigError::InvalidParameter { param, reason } => {
                write!(f, "Invalid parameter '{}': {}", param, reason)
            }
            ConfigError::MissingRequired(param) => write!(f, "Missing required parameter: {}", param),
            ConfigError::FileNotFound(path) => write!(f, "Configuration file not found: {}", path),
            ConfigError::InvalidFormat(msg) => write!(f, "Invalid configuration format: {}", msg),
            ConfigError::ValidationFailed(msg) => write!(f, "Configuration validation failed: {}", msg),
        }
    }
}

impl fmt::Display for DeviceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceError::NoDeviceAvailable => write!(f, "No suitable device available"),
            DeviceError::DeviceTypeUnavailable(device_type) => {
                write!(f, "Device type '{}' not available", device_type)
            }
            DeviceError::InitializationFailed(msg) => write!(f, "Device initialization failed: {}", msg),
            DeviceError::FeatureNotEnabled(feature) => {
                write!(f, "Device feature '{}' not enabled", feature)
            }
        }
    }
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelError::FileNotFound(path) => write!(f, "Model file not found: {}", path),
            ModelError::IncompatibleVersion { expected, found } => {
                write!(f, "Incompatible model version: expected {}, found {}", expected, found)
            }
            ModelError::MissingComponent(component) => write!(f, "Missing model component: {}", component),
            ModelError::LoadingFailed(msg) => write!(f, "Model loading failed: {}", msg),
            ModelError::ArchitectureMismatch(msg) => write!(f, "Model architecture mismatch: {}", msg),
        }
    }
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenizerError::LoadingFailed(msg) => write!(f, "Tokenizer loading failed: {}", msg),
            TokenizerError::TokenizationFailed(msg) => write!(f, "Tokenization failed: {}", msg),
            TokenizerError::UnsupportedType(tokenizer_type) => {
                write!(f, "Unsupported tokenizer type: {}", tokenizer_type)
            }
            TokenizerError::VocabError(msg) => write!(f, "Vocabulary error: {}", msg),
        }
    }
}

impl fmt::Display for HubError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HubError::AuthenticationFailed => write!(f, "HuggingFace authentication failed"),
            HubError::ModelNotFound(model_id) => write!(f, "Model '{}' not found on HuggingFace Hub", model_id),
            HubError::DownloadFailed(msg) => write!(f, "Download failed: {}", msg),
            HubError::NetworkError(msg) => write!(f, "Network error: {}", msg),
            HubError::InvalidRepository(repo) => write!(f, "Invalid repository: {}", repo),
        }
    }
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorError::ShapeMismatch { expected, found } => {
                write!(f, "Shape mismatch: expected {}, found {}", expected, found)
            }
            TensorError::UnsupportedOperation(op) => write!(f, "Unsupported operation: {}", op),
            TensorError::ComputationFailed(msg) => write!(f, "Computation failed: {}", msg),
            TensorError::OutOfMemory => write!(f, "Out of memory"),
            TensorError::TypeConversionFailed(msg) => write!(f, "Type conversion failed: {}", msg),
        }
    }
}

impl std::error::Error for TurboCALMError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            TurboCALMError::Io(e) => Some(e),
            TurboCALMError::Candle(e) => Some(e),
            TurboCALMError::Json(e) => Some(e),
            _ => None,
        }
    }
}

impl std::error::Error for ConfigError {}
impl std::error::Error for DeviceError {}
impl std::error::Error for ModelError {}
impl std::error::Error for TokenizerError {}
impl std::error::Error for HubError {}
impl std::error::Error for TensorError {}

// Conversions from common error types
impl From<std::io::Error> for TurboCALMError {
    fn from(err: std::io::Error) -> Self {
        TurboCALMError::Io(err)
    }
}

impl From<candle_core::Error> for TurboCALMError {
    fn from(err: candle_core::Error) -> Self {
        TurboCALMError::Candle(err)
    }
}

impl From<serde_json::Error> for TurboCALMError {
    fn from(err: serde_json::Error) -> Self {
        TurboCALMError::Json(err)
    }
}

impl From<ConfigError> for TurboCALMError {
    fn from(err: ConfigError) -> Self {
        TurboCALMError::Config(err)
    }
}

impl From<DeviceError> for TurboCALMError {
    fn from(err: DeviceError) -> Self {
        TurboCALMError::Device(err)
    }
}

impl From<ModelError> for TurboCALMError {
    fn from(err: ModelError) -> Self {
        TurboCALMError::Model(err)
    }
}

impl From<TokenizerError> for TurboCALMError {
    fn from(err: TokenizerError) -> Self {
        TurboCALMError::Tokenizer(err)
    }
}

impl From<HubError> for TurboCALMError {
    fn from(err: HubError) -> Self {
        TurboCALMError::Hub(err)
    }
}

impl From<TensorError> for TurboCALMError {
    fn from(err: TensorError) -> Self {
        TurboCALMError::Tensor(err)
    }
}

/// Result type for turbocalm-core operations
pub type Result<T> = std::result::Result<T, TurboCALMError>;

/// Convenience macros for creating errors
#[macro_export]
macro_rules! config_error {
    ($param:expr, $reason:expr) => {
        $crate::error::TurboCALMError::Config($crate::error::ConfigError::InvalidParameter {
            param: $param.to_string(),
            reason: $reason.to_string(),
        })
    };
}

#[macro_export]
macro_rules! tensor_error {
    (shape_mismatch, $expected:expr, $found:expr) => {
        $crate::error::TurboCALMError::Tensor($crate::error::TensorError::ShapeMismatch {
            expected: $expected.to_string(),
            found: $found.to_string(),
        })
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let config_err = ConfigError::MissingRequired("vocab_size".to_string());
        assert_eq!(format!("{}", config_err), "Missing required parameter: vocab_size");

        let main_err = TurboCALMError::Config(config_err);
        assert!(format!("{}", main_err).contains("Configuration error"));
    }

    #[test]
    fn test_error_conversions() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let turbo_err: TurboCALMError = io_err.into();
        assert!(matches!(turbo_err, TurboCALMError::Io(_)));
    }

    #[test]
    fn test_result_type() {
        fn dummy_function() -> Result<i32> {
            Ok(42)
        }

        assert_eq!(dummy_function().unwrap(), 42);
    }

    #[test]
    fn test_error_macros() {
        let err = config_error!("test_param", "test reason");
        assert!(matches!(err, TurboCALMError::Config(ConfigError::InvalidParameter { .. })));

        let tensor_err = tensor_error!(shape_mismatch, "[2, 3]", "[3, 4]");
        assert!(matches!(tensor_err, TurboCALMError::Tensor(TensorError::ShapeMismatch { .. })));
    }
}