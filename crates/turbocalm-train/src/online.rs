use std::path::Path;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, Result};
use candle_core::Device;
use turbocalm_models::CalmAutoencoderConfig;

use crate::checkpoint::CheckpointInfo;
use crate::corpus::{build_pairs_from_entries, CorpusEntry};
use crate::{Trainer, TrainingConfig};

#[derive(Debug, Clone)]
pub struct OnlineLearningConfig {
    pub buffer_size: usize,
    pub mini_epochs: usize,
}

impl Default for OnlineLearningConfig {
    fn default() -> Self {
        Self {
            buffer_size: 50,
            mini_epochs: 2,
        }
    }
}

#[derive(Debug, Clone)]
pub struct OnlineAddResult {
    pub buffered: usize,
    pub threshold: usize,
    pub trained: bool,
    pub pair_count: usize,
    pub epochs_ran: usize,
    pub checkpoint: Option<CheckpointInfo>,
}

struct OnlineState {
    trainer: Trainer,
    buffer: Vec<CorpusEntry>,
}

pub struct OnlineLearner {
    state: Mutex<OnlineState>,
    config: OnlineLearningConfig,
}

impl OnlineLearner {
    pub fn new(
        model_config: CalmAutoencoderConfig,
        mut training_config: TrainingConfig,
        device: Device,
        config: OnlineLearningConfig,
    ) -> Result<Self> {
        validate_online_config(&config)?;
        training_config.min_corpus_size = 1;
        let trainer = Trainer::new(model_config, training_config, device)?;
        Ok(Self {
            state: Mutex::new(OnlineState {
                trainer,
                buffer: Vec::new(),
            }),
            config,
        })
    }

    pub fn from_checkpoint<P: AsRef<Path>>(
        checkpoint_path: P,
        model_config: CalmAutoencoderConfig,
        mut training_config: TrainingConfig,
        device: Device,
        config: OnlineLearningConfig,
    ) -> Result<Self> {
        validate_online_config(&config)?;
        training_config.min_corpus_size = 1;
        let trainer =
            Trainer::from_checkpoint(checkpoint_path, model_config, training_config, device)?;
        Ok(Self {
            state: Mutex::new(OnlineState {
                trainer,
                buffer: Vec::new(),
            }),
            config,
        })
    }

    pub fn add_text(&self, text: &str, category: &str) -> Result<OnlineAddResult> {
        if text.trim().is_empty() {
            bail!("online learning text must not be empty")
        }
        if category.trim().is_empty() {
            bail!("online learning category must not be empty")
        }

        let mut state = self.state.lock().unwrap();
        state.buffer.push(CorpusEntry {
            text: text.trim().to_string(),
            category: category.trim().to_string(),
            timestamp: now_timestamp(),
            source: "online".to_string(),
        });

        if state.buffer.len() < self.config.buffer_size {
            return Ok(OnlineAddResult {
                buffered: state.buffer.len(),
                threshold: self.config.buffer_size,
                trained: false,
                pair_count: 0,
                epochs_ran: 0,
                checkpoint: None,
            });
        }

        let batch = state.buffer.drain(..).collect::<Vec<_>>();
        let corpus = build_pairs_from_entries(&batch);
        let pair_count = corpus.pairs.len();
        if pair_count == 0 {
            return Ok(OnlineAddResult {
                buffered: 0,
                threshold: self.config.buffer_size,
                trained: false,
                pair_count,
                epochs_ran: 0,
                checkpoint: None,
            });
        }

        let summary = state
            .trainer
            .train_with_epoch_limit(&corpus, self.config.mini_epochs)?;
        Ok(OnlineAddResult {
            buffered: state.buffer.len(),
            threshold: self.config.buffer_size,
            trained: true,
            pair_count,
            epochs_ran: summary.epoch_losses.len(),
            checkpoint: summary.best_checkpoint,
        })
    }

    pub fn buffered_len(&self) -> usize {
        self.state.lock().unwrap().buffer.len()
    }

    pub fn model_config(&self) -> CalmAutoencoderConfig {
        self.state.lock().unwrap().trainer.model_config().clone()
    }

    pub fn embed_texts_pooled(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.state.lock().unwrap().trainer.embed_texts_pooled(texts)
    }

    pub fn embed_texts_chunked(&self, texts: &[String]) -> Result<Vec<Vec<Vec<f32>>>> {
        self.state
            .lock()
            .unwrap()
            .trainer
            .embed_texts_chunked(texts)
    }
}

fn validate_online_config(config: &OnlineLearningConfig) -> Result<()> {
    if config.buffer_size == 0 {
        bail!("online buffer_size must be greater than zero")
    }
    if !(1..=3).contains(&config.mini_epochs) {
        bail!("online mini_epochs must be between 1 and 3")
    }
    Ok(())
}

fn now_timestamp() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn auto_trains_when_buffer_reaches_threshold() -> Result<()> {
        let checkpoint_dir = temp_dir("online");
        let training_config = TrainingConfig {
            checkpoint_dir: checkpoint_dir.clone(),
            min_corpus_size: 1,
            max_epochs: 3,
            eval_interval: 1,
            patience: 3,
            ..TrainingConfig::default()
        };
        let learner = OnlineLearner::new(
            small_model_config(),
            training_config,
            Device::Cpu,
            OnlineLearningConfig {
                buffer_size: 3,
                mini_epochs: 1,
            },
        )?;

        assert!(!learner.add_text("breathing calm one", "breathing")?.trained);
        assert!(!learner.add_text("breathing calm two", "breathing")?.trained);
        let result = learner.add_text("breathing calm three", "breathing")?;

        assert!(result.trained);
        assert!(result.pair_count > 0);
        assert!(result.checkpoint.is_some());
        std::fs::remove_dir_all(checkpoint_dir)?;
        Ok(())
    }

    fn small_model_config() -> CalmAutoencoderConfig {
        CalmAutoencoderConfig {
            vocab_size: 512,
            hidden_size: 16,
            intermediate_size: 32,
            latent_size: 8,
            num_encoder_layers: 2,
            num_decoder_layers: 2,
            patch_size: 4,
            tie_word_embeddings: true,
            ..Default::default()
        }
    }

    fn temp_dir(label: &str) -> std::path::PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "turbocalm-train-online-{label}-{}-{unique}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path).unwrap();
        path
    }
}
