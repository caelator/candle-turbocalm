use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use candle_core::{backprop::GradStore, DType, Device, Tensor, Var};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use turbocalm_models::{CalmAutoencoder, CalmAutoencoderConfig};

use crate::checkpoint::{
    checkpoint_path_in_dir, next_checkpoint_version_in_dir, save_checkpoint,
    save_checkpoint_config, CheckpointInfo,
};
use crate::embedding::{encode_text_batch, preseed_embeddings, TOKEN_OFFSET};
use crate::loss::{self, DEFAULT_TEMPERATURE};
use crate::pairs::{generate_epoch_batches, Corpus};

const DEFAULT_LOG_EVERY_BATCHES: usize = 10;
const MAX_GRAD_NORM: f64 = 1.0;

#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub batch_size: usize,
    pub lr: f64,
    pub weight_decay: f64,
    pub temperature: f64,
    pub max_epochs: usize,
    pub eval_interval: usize,
    pub patience: usize,
    pub checkpoint_dir: PathBuf,
    pub min_corpus_size: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            lr: 1e-4,
            weight_decay: 0.01,
            temperature: DEFAULT_TEMPERATURE,
            max_epochs: 20,
            eval_interval: 2,
            patience: 3,
            checkpoint_dir: crate::checkpoint::default_checkpoint_dir()
                .unwrap_or_else(|_| PathBuf::from(".turbocalm/trained")),
            min_corpus_size: 200,
        }
    }
}

#[derive(Debug)]
pub struct TrainingSummary {
    pub epoch_losses: Vec<f32>,
    pub best_loss: f32,
    pub best_checkpoint: Option<CheckpointInfo>,
    pub stopped_early: bool,
}

pub struct Trainer {
    pub model: CalmAutoencoder,
    pub varmap: VarMap,
    pub optimizer: AdamW,
    device: Device,
    config: TrainingConfig,
    trainable_vars: Vec<Var>,
}

impl Trainer {
    pub fn new(
        model_config: CalmAutoencoderConfig,
        config: TrainingConfig,
        device: Device,
    ) -> Result<Self> {
        validate_config(&config)?;
        validate_model_config(&model_config)?;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        preseed_embeddings(&vb, &model_config)?;

        let model = CalmAutoencoder::load(vb, model_config)
            .context("failed to build trainable CALM autoencoder")?;
        let trainable_vars = varmap.all_vars();
        let optimizer = AdamW::new(
            trainable_vars.clone(),
            ParamsAdamW {
                lr: config.lr,
                weight_decay: config.weight_decay,
                ..ParamsAdamW::default()
            },
        )
        .context("failed to create AdamW optimizer")?;

        Ok(Self {
            model,
            varmap,
            optimizer,
            device,
            config,
            trainable_vars,
        })
    }

    pub fn config(&self) -> &TrainingConfig {
        &self.config
    }

    pub fn train_epoch(&mut self, corpus: &Corpus) -> Result<f32> {
        if corpus.pairs.is_empty() {
            bail!("cannot train on an empty corpus")
        }
        if self.config.batch_size == 0 {
            bail!("batch_size must be greater than zero")
        }

        let batches = generate_epoch_batches(corpus, self.config.batch_size);
        if batches.is_empty() {
            bail!("no training batches were generated")
        }

        let mut total_loss = 0f32;
        for (batch_index, batch) in batches.iter().enumerate() {
            let anchor_tokens = self.encode_text_batch(
                &batch
                    .iter()
                    .map(|pair| pair.anchor.clone())
                    .collect::<Vec<_>>(),
            )?;
            let positive_tokens = self.encode_text_batch(
                &batch
                    .iter()
                    .map(|pair| pair.positive.clone())
                    .collect::<Vec<_>>(),
            )?;

            let anchor_embeddings = self
                .model
                .encode_pooled(&anchor_tokens)
                .context("failed to encode anchor batch")?;
            let positive_embeddings = self
                .model
                .encode_pooled(&positive_tokens)
                .context("failed to encode positive batch")?;

            let loss = loss::nt_xent_loss(
                &anchor_embeddings,
                &positive_embeddings,
                self.config.temperature,
            )?;
            let loss_value = scalar_f32(&loss)?;
            let mut grads = loss
                .backward()
                .context("failed to backprop training loss")?;
            let grad_norm = clip_gradients(&mut grads, &self.trainable_vars, MAX_GRAD_NORM)?;

            self.optimizer
                .step(&grads)
                .context("failed to apply AdamW optimizer step")?;

            total_loss += loss_value;

            let should_log = (batch_index + 1) % DEFAULT_LOG_EVERY_BATCHES == 0
                || batch_index + 1 == batches.len();
            if should_log {
                println!(
                    "batch {}/{} loss={:.6} grad_norm={:.4}",
                    batch_index + 1,
                    batches.len(),
                    loss_value,
                    grad_norm
                );
            }
        }

        Ok(total_loss / batches.len() as f32)
    }

    pub fn train(&mut self, corpus: &Corpus) -> Result<TrainingSummary> {
        if corpus.pairs.len() < self.config.min_corpus_size {
            bail!(
                "corpus too small for training: {} pairs < minimum {}",
                corpus.pairs.len(),
                self.config.min_corpus_size
            )
        }

        let version = next_checkpoint_version_in_dir(&self.config.checkpoint_dir)?;
        let checkpoint_path = checkpoint_path_in_dir(&self.config.checkpoint_dir, version)?;
        let mut best_checkpoint = None;
        let mut best_loss = f32::INFINITY;
        let mut epoch_losses = Vec::with_capacity(self.config.max_epochs);
        let mut plateau_count = 0usize;
        let mut stopped_early = false;

        for epoch in 1..=self.config.max_epochs {
            let epoch_loss = self.train_epoch(corpus)?;
            println!("epoch {epoch} average_loss={epoch_loss:.6}");
            epoch_losses.push(epoch_loss);

            let should_evaluate = epoch == 1
                || epoch == self.config.max_epochs
                || epoch % self.config.eval_interval == 0;
            if !should_evaluate {
                continue;
            }

            if epoch_loss + 1e-6 < best_loss {
                best_loss = epoch_loss;
                plateau_count = 0;
                let checkpoint = save_checkpoint(&self.varmap, &checkpoint_path, version)
                    .with_context(|| {
                        format!(
                            "failed to save training checkpoint to {}",
                            checkpoint_path.display()
                        )
                    })?;
                save_checkpoint_config(self.model.config(), &checkpoint.path).with_context(
                    || {
                        format!(
                            "failed to save checkpoint config for {}",
                            checkpoint.path.display()
                        )
                    },
                )?;
                best_checkpoint = Some(checkpoint);
            } else {
                plateau_count += 1;
                if plateau_count >= self.config.patience {
                    stopped_early = true;
                    break;
                }
            }
        }

        if !best_loss.is_finite() {
            bail!("training finished without producing a finite loss")
        }

        Ok(TrainingSummary {
            epoch_losses,
            best_loss,
            best_checkpoint,
            stopped_early,
        })
    }

    fn encode_text_batch(&self, texts: &[String]) -> Result<Tensor> {
        encode_text_batch(texts, self.model.config(), &self.device)
    }
}

fn validate_config(config: &TrainingConfig) -> Result<()> {
    if config.batch_size == 0 {
        bail!("batch_size must be greater than zero")
    }
    if config.lr <= 0.0 {
        bail!("lr must be positive")
    }
    if config.temperature <= 0.0 {
        bail!("temperature must be positive")
    }
    if config.max_epochs == 0 {
        bail!("max_epochs must be greater than zero")
    }
    if config.eval_interval == 0 {
        bail!("eval_interval must be greater than zero")
    }
    if config.patience == 0 {
        bail!("patience must be greater than zero")
    }
    Ok(())
}

fn validate_model_config(config: &CalmAutoencoderConfig) -> Result<()> {
    if config.vocab_size <= TOKEN_OFFSET as usize {
        bail!("model vocab_size must be greater than {TOKEN_OFFSET}")
    }
    Ok(())
}

fn clip_gradients(grads: &mut GradStore, vars: &[Var], max_norm: f64) -> Result<f64> {
    let mut total_norm_sq = 0f64;

    for var in vars {
        if let Some(grad) = grads.get(var.as_tensor()) {
            let norm_sq = grad
                .sqr()
                .context("failed to square gradient for clipping")?
                .sum_all()
                .context("failed to sum gradient norm")?;
            total_norm_sq += scalar_f32(&norm_sq)? as f64;
        }
    }

    let total_norm = total_norm_sq.sqrt();
    if total_norm <= max_norm || total_norm == 0.0 {
        return Ok(total_norm);
    }

    let clip_scale = max_norm / (total_norm + 1e-12);
    for var in vars {
        if let Some(grad) = grads.remove(var.as_tensor()) {
            let clipped = (&grad * clip_scale).context("failed to scale gradient for clipping")?;
            grads.insert(var.as_tensor(), clipped);
        }
    }

    Ok(total_norm)
}

fn scalar_f32(tensor: &Tensor) -> Result<f32> {
    tensor
        .to_device(&Device::Cpu)
        .context("failed to copy scalar tensor to CPU")?
        .to_scalar::<f32>()
        .context("failed to read scalar tensor")
}
