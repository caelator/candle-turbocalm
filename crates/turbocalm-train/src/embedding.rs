use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::Path;

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Init, VarBuilder, VarMap};
use turbocalm_models::{CalmAutoencoder, CalmAutoencoderConfig};

pub const MAX_TEXT_TOKENS: usize = 128;
pub const TOKEN_OFFSET: u32 = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingMode {
    Pooled,
    Chunked,
}

pub struct EmbeddingModel {
    model: CalmAutoencoder,
    device: Device,
}

impl EmbeddingModel {
    pub fn new(model: CalmAutoencoder, device: Device) -> Self {
        Self { model, device }
    }

    pub fn random(config: CalmAutoencoderConfig, device: Device) -> Result<Self> {
        let model = build_random_autoencoder(&config, &device)?.0;
        Ok(Self::new(model, device))
    }

    pub fn from_checkpoint<P: AsRef<Path>>(
        path: P,
        config: CalmAutoencoderConfig,
        device: Device,
    ) -> Result<Self> {
        let model = CalmAutoencoder::from_safetensors(path, config, DType::F32, &device)?;
        Ok(Self::new(model, device))
    }

    pub fn config(&self) -> &CalmAutoencoderConfig {
        self.model.config()
    }

    pub fn embedding_dim(&self) -> usize {
        self.model.config().latent_size
    }

    pub fn encode_input_ids_chunked(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.model.encode_chunked(input_ids)
    }

    pub fn encode_input_ids_pooled(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.model.encode_pooled(input_ids)
    }

    pub fn embed_texts_pooled(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let input_ids = encode_text_batch(texts, self.model.config(), &self.device)?;
        let embeddings = self
            .model
            .encode_pooled(&input_ids)
            .context("failed to encode pooled embeddings")?;
        tensor_to_rows(&embeddings)
    }

    pub fn embed_texts_chunked(&self, texts: &[String]) -> Result<Vec<Vec<Vec<f32>>>> {
        let input_ids = encode_text_batch(texts, self.model.config(), &self.device)?;
        let embeddings = self
            .model
            .encode_chunked(&input_ids)
            .context("failed to encode chunked embeddings")?;
        tensor_to_chunks(&embeddings)
    }
}

pub fn build_random_autoencoder(
    config: &CalmAutoencoderConfig,
    device: &Device,
) -> Result<(CalmAutoencoder, VarMap)> {
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    preseed_embeddings(&vb, config)?;
    let model = CalmAutoencoder::load(vb, config.clone())
        .context("failed to build random CALM autoencoder")?;
    Ok((model, varmap))
}

pub fn preseed_embeddings(vb: &VarBuilder<'_>, config: &CalmAutoencoderConfig) -> Result<()> {
    let init = Init::Randn {
        mean: 0.0,
        stdev: config.initializer_range,
    };

    vb.pp("encoder")
        .get_with_hints(
            (config.vocab_size, config.hidden_size),
            "embed_tokens.weight",
            init,
        )
        .context("failed to initialize encoder embeddings")?;

    if !config.tie_word_embeddings {
        vb.pp("decoder")
            .get_with_hints(
                (config.vocab_size, config.hidden_size),
                "lm_head.weight",
                init,
            )
            .context("failed to initialize decoder output embeddings")?;
    }

    Ok(())
}

pub fn encode_text_batch(
    texts: &[String],
    config: &CalmAutoencoderConfig,
    device: &Device,
) -> Result<Tensor> {
    let seq_len = batch_sequence_len(texts, config)?;
    let token_ids = texts
        .iter()
        .flat_map(|text| encode_text(text, config, seq_len))
        .collect::<Vec<_>>();

    Tensor::from_vec(token_ids, (texts.len(), seq_len), device)
        .context("failed to build token batch tensor")
}

pub fn token_count_for_text(text: &str, config: &CalmAutoencoderConfig) -> usize {
    let token_count = text.split_whitespace().count();
    if token_count == 0 {
        text.chars().count().min(MAX_TEXT_TOKENS.saturating_sub(2)) + 2
    } else {
        (token_count + 2)
            .min(config.max_position_embeddings)
            .min(MAX_TEXT_TOKENS)
    }
}

fn batch_sequence_len(texts: &[String], config: &CalmAutoencoderConfig) -> Result<usize> {
    if texts.is_empty() {
        bail!("cannot encode an empty text batch")
    }

    Ok(texts
        .iter()
        .map(|text| token_count_for_text(text, config))
        .max()
        .unwrap_or(2)
        .max(2))
}

fn encode_text(text: &str, config: &CalmAutoencoderConfig, seq_len: usize) -> Vec<u32> {
    let usable_vocab = (config.vocab_size as u32)
        .saturating_sub(TOKEN_OFFSET)
        .max(1);
    let pad_token = config.pad_token_id.unwrap_or_default();
    let budget = seq_len.saturating_sub(2);

    let mut ids = Vec::with_capacity(seq_len);
    ids.push(config.bos_token_id);

    let mut appended = 0usize;
    for token in text.split_whitespace() {
        if appended >= budget {
            break;
        }
        ids.push(TOKEN_OFFSET + stable_token_hash(token) % usable_vocab);
        appended += 1;
    }

    if appended == 0 && budget > 0 {
        for ch in text.chars() {
            if appended >= budget {
                break;
            }
            let mut buffer = [0u8; 4];
            let token = ch.encode_utf8(&mut buffer);
            ids.push(TOKEN_OFFSET + stable_token_hash(token) % usable_vocab);
            appended += 1;
        }
    }

    ids.push(config.eos_token_id);
    ids.resize(seq_len, pad_token);
    ids
}

fn stable_token_hash(token: &str) -> u32 {
    let mut hasher = DefaultHasher::new();
    token.to_lowercase().hash(&mut hasher);
    hasher.finish() as u32
}

fn tensor_to_rows(tensor: &Tensor) -> Result<Vec<Vec<f32>>> {
    let (rows, dims) = tensor.dims2().context("expected rank-2 embedding tensor")?;
    let values = tensor
        .to_device(&Device::Cpu)
        .context("failed to move embeddings to CPU")?
        .flatten_all()
        .context("failed to flatten pooled embeddings")?
        .to_vec1::<f32>()
        .context("failed to extract pooled embedding values")?;

    Ok(values
        .chunks(dims)
        .take(rows)
        .map(|chunk| chunk.to_vec())
        .collect())
}

fn tensor_to_chunks(tensor: &Tensor) -> Result<Vec<Vec<Vec<f32>>>> {
    let (batch, patches, dims) = tensor
        .dims3()
        .context("expected rank-3 chunked embedding tensor")?;
    let values = tensor
        .to_device(&Device::Cpu)
        .context("failed to move chunked embeddings to CPU")?
        .flatten_all()
        .context("failed to flatten chunked embeddings")?
        .to_vec1::<f32>()
        .context("failed to extract chunked embedding values")?;

    let mut offset = 0usize;
    let mut batches = Vec::with_capacity(batch);
    for _ in 0..batch {
        let mut chunk_vectors = Vec::with_capacity(patches);
        for _ in 0..patches {
            let end = offset + dims;
            chunk_vectors.push(values[offset..end].to_vec());
            offset = end;
        }
        batches.push(chunk_vectors);
    }
    Ok(batches)
}
