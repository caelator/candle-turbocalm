use crate::error::{Result, TokenizerError, TurboCALMError};
use hf_hub::api::sync::Api;
use std::path::Path;
use tokenizers::Tokenizer;
use tracing::{debug, info};

/// Supported tokenizer types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizerType {
    /// Llama/Llama2/Llama3 tokenizer
    Llama,
    /// Generic BPE tokenizer
    Bpe,
    /// SentencePiece tokenizer
    SentencePiece,
}

impl TokenizerType {
    /// Get the expected tokenizer filename for this type
    pub fn default_filename(&self) -> &'static str {
        match self {
            TokenizerType::Llama => "tokenizer.json",
            TokenizerType::Bpe => "tokenizer.json",
            TokenizerType::SentencePiece => "tokenizer.model",
        }
    }
}

/// Tokenizer loader with HuggingFace Hub integration
pub struct TokenizerLoader {
    api: Api,
}

impl TokenizerLoader {
    /// Create a new tokenizer loader
    pub fn new() -> Result<Self> {
        let api = Api::new().map_err(|e| {
            TurboCALMError::Tokenizer(TokenizerError::LoadingFailed(format!(
                "Failed to initialize HF API: {}",
                e
            )))
        })?;

        Ok(Self { api })
    }

    /// Load tokenizer from HuggingFace model repository
    pub fn load_from_hub(
        &self,
        model_id: &str,
        tokenizer_type: TokenizerType,
    ) -> Result<Tokenizer> {
        info!("Loading tokenizer from HuggingFace Hub: {}", model_id);

        let repo = self.api.model(model_id.to_string());
        let filename = tokenizer_type.default_filename();

        debug!("Downloading tokenizer file: {}", filename);
        let tokenizer_path = repo.get(filename).map_err(|e| {
            TurboCALMError::Tokenizer(TokenizerError::LoadingFailed(format!(
                "Failed to download {} from {}: {}",
                filename, model_id, e
            )))
        })?;

        self.load_from_file(&tokenizer_path, tokenizer_type)
    }

    /// Load tokenizer from local file
    pub fn load_from_file<P: AsRef<Path>>(
        &self,
        path: P,
        tokenizer_type: TokenizerType,
    ) -> Result<Tokenizer> {
        let path = path.as_ref();
        info!("Loading tokenizer from file: {}", path.display());

        if !path.exists() {
            return Err(TurboCALMError::Tokenizer(TokenizerError::LoadingFailed(
                format!("Tokenizer file not found: {}", path.display()),
            )));
        }

        match tokenizer_type {
            TokenizerType::Llama | TokenizerType::Bpe => Tokenizer::from_file(path).map_err(|e| {
                TurboCALMError::Tokenizer(TokenizerError::LoadingFailed(format!(
                    "Failed to load tokenizer from {}: {}",
                    path.display(),
                    e
                )))
            }),
            TokenizerType::SentencePiece => {
                // For SentencePiece, we'd typically need to convert it to a JSON tokenizer
                // This is a simplified implementation - in practice you might need
                // to handle SentencePiece differently
                Err(TurboCALMError::Tokenizer(TokenizerError::UnsupportedType(
                    "SentencePiece tokenizer loading not fully implemented".to_string(),
                )))
            }
        }
    }

    /// Load the default Llama3 tokenizer
    pub fn load_llama3(&self) -> Result<Tokenizer> {
        // Llama 3 model ID on HuggingFace Hub
        const LLAMA3_MODEL_ID: &str = "meta-llama/Meta-Llama-3-8B";
        self.load_from_hub(LLAMA3_MODEL_ID, TokenizerType::Llama)
    }

    /// Load Llama2 tokenizer
    pub fn load_llama2(&self) -> Result<Tokenizer> {
        const LLAMA2_MODEL_ID: &str = "meta-llama/Llama-2-7b-hf";
        self.load_from_hub(LLAMA2_MODEL_ID, TokenizerType::Llama)
    }

    /// Load original Llama tokenizer
    pub fn load_llama(&self) -> Result<Tokenizer> {
        const LLAMA_MODEL_ID: &str = "huggyllama/llama-7b";
        self.load_from_hub(LLAMA_MODEL_ID, TokenizerType::Llama)
    }
}

impl Default for TokenizerLoader {
    fn default() -> Self {
        Self::new().expect("Failed to create default tokenizer loader")
    }
}

/// Tokenizer utilities and extensions
pub struct TokenizerUtils;

impl TokenizerUtils {
    /// Get tokenizer vocabulary size
    pub fn vocab_size(tokenizer: &Tokenizer) -> usize {
        tokenizer.get_vocab_size(true)
    }

    /// Get special token IDs
    pub fn get_special_tokens(tokenizer: &Tokenizer) -> SpecialTokens {
        let special_tokens = tokenizer.get_vocab(true);

        SpecialTokens {
            bos_token_id: special_tokens
                .get("<s>")
                .copied()
                .or_else(|| special_tokens.get("<|begin_of_text|>").copied()),
            eos_token_id: special_tokens
                .get("</s>")
                .copied()
                .or_else(|| special_tokens.get("<|end_of_text|>").copied()),
            pad_token_id: special_tokens.get("<pad>").copied(),
            unk_token_id: special_tokens.get("<unk>").copied(),
        }
    }

    /// Encode text to token IDs
    pub fn encode(tokenizer: &Tokenizer, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        tokenizer
            .encode(text, add_special_tokens)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|e| {
                TurboCALMError::Tokenizer(TokenizerError::TokenizationFailed(format!(
                    "Failed to encode text: {}",
                    e
                )))
            })
    }

    /// Decode token IDs to text
    pub fn decode(
        tokenizer: &Tokenizer,
        token_ids: &[u32],
        skip_special_tokens: bool,
    ) -> Result<String> {
        tokenizer
            .decode(token_ids, skip_special_tokens)
            .map_err(|e| {
                TurboCALMError::Tokenizer(TokenizerError::TokenizationFailed(format!(
                    "Failed to decode tokens: {}",
                    e
                )))
            })
    }

    /// Batch encode multiple texts
    pub fn encode_batch(
        tokenizer: &Tokenizer,
        texts: &[String],
        add_special_tokens: bool,
    ) -> Result<Vec<Vec<u32>>> {
        tokenizer
            .encode_batch(
                texts.iter().map(|s| s.as_str()).collect(),
                add_special_tokens,
            )
            .map(|encodings| {
                encodings
                    .into_iter()
                    .map(|enc| enc.get_ids().to_vec())
                    .collect()
            })
            .map_err(|e| {
                TurboCALMError::Tokenizer(TokenizerError::TokenizationFailed(format!(
                    "Failed to encode batch: {}",
                    e
                )))
            })
    }
}

/// Special token IDs for a tokenizer
#[derive(Debug, Clone)]
pub struct SpecialTokens {
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub pad_token_id: Option<u32>,
    pub unk_token_id: Option<u32>,
}

impl SpecialTokens {
    /// Get BOS token ID, with fallback value
    pub fn bos_token_id_or(&self, fallback: u32) -> u32 {
        self.bos_token_id.unwrap_or(fallback)
    }

    /// Get EOS token ID, with fallback value
    pub fn eos_token_id_or(&self, fallback: u32) -> u32 {
        self.eos_token_id.unwrap_or(fallback)
    }

    /// Get PAD token ID, using EOS as fallback if PAD not available
    pub fn pad_token_id_or_eos(&self) -> Option<u32> {
        self.pad_token_id.or(self.eos_token_id)
    }
}

/// Convenience functions for common tokenizer operations
pub mod convenience {
    use super::*;

    /// Quick function to load Llama3 tokenizer
    pub fn load_llama3_tokenizer() -> Result<Tokenizer> {
        TokenizerLoader::new()?.load_llama3()
    }

    /// Quick function to load any tokenizer from HuggingFace
    pub fn load_tokenizer(model_id: &str, tokenizer_type: TokenizerType) -> Result<Tokenizer> {
        TokenizerLoader::new()?.load_from_hub(model_id, tokenizer_type)
    }

    /// Quick encode function
    pub fn encode_text(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>> {
        TokenizerUtils::encode(tokenizer, text, true)
    }

    /// Quick decode function
    pub fn decode_tokens(tokenizer: &Tokenizer, tokens: &[u32]) -> Result<String> {
        TokenizerUtils::decode(tokenizer, tokens, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_type() {
        assert_eq!(TokenizerType::Llama.default_filename(), "tokenizer.json");
        assert_eq!(TokenizerType::Bpe.default_filename(), "tokenizer.json");
        assert_eq!(
            TokenizerType::SentencePiece.default_filename(),
            "tokenizer.model"
        );
    }

    #[test]
    fn test_special_tokens_fallback() {
        let special_tokens = SpecialTokens {
            bos_token_id: Some(1),
            eos_token_id: Some(2),
            pad_token_id: None,
            unk_token_id: Some(0),
        };

        assert_eq!(special_tokens.bos_token_id_or(999), 1);
        assert_eq!(special_tokens.eos_token_id_or(999), 2);
        assert_eq!(special_tokens.pad_token_id_or_eos(), Some(2)); // Falls back to EOS

        let no_special_tokens = SpecialTokens {
            bos_token_id: None,
            eos_token_id: None,
            pad_token_id: None,
            unk_token_id: None,
        };

        assert_eq!(no_special_tokens.bos_token_id_or(999), 999);
        assert_eq!(no_special_tokens.eos_token_id_or(999), 999);
        assert_eq!(no_special_tokens.pad_token_id_or_eos(), None);
    }

    #[test]
    fn test_tokenizer_loader_creation() {
        // This test might fail if there's no network connection or HF API issues
        // In a real test environment, you'd mock the API
        let result = TokenizerLoader::new();
        // We can't guarantee this works in all environments, so we just check it doesn't panic
        match result {
            Ok(_) => println!("TokenizerLoader created successfully"),
            Err(e) => println!(
                "TokenizerLoader creation failed (expected in some environments): {}",
                e
            ),
        }
    }
}
