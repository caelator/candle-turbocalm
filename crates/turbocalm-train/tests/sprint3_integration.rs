use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use candle_core::Device;
use turbocalm_models::CalmAutoencoderConfig;
use turbocalm_train::checkpoint::latest_checkpoint_path_in_dir;
use turbocalm_train::embedding::encode_text_batch;
use turbocalm_train::pairs::{Corpus, CorpusMetadata, TrainingPair};
use turbocalm_train::{run_eval, EmbeddingModel, EvalCorpus, EvalPair, Trainer, TrainingConfig};
use turbocalm_triumvirate::EmbeddingEngine;

#[test]
fn eval_recall_is_positive_after_training() -> Result<()> {
    let temp_dir = TempDirGuard::new("eval")?;
    let (config, checkpoint_path) = train_checkpoint(temp_dir.path(), 32)?;
    let model = EmbeddingModel::from_checkpoint(&checkpoint_path, config, Device::Cpu)?;
    let metrics = run_eval(&model, &synthetic_eval_corpus())?;

    assert!(
        metrics.recall_at_5 > 0.0,
        "expected positive recall@5 after training, got {}",
        metrics.recall_at_5
    );
    assert!(
        metrics.mrr > 0.0,
        "expected positive MRR, got {}",
        metrics.mrr
    );
    Ok(())
}

#[test]
fn trained_checkpoint_is_loaded_by_embedding_engine() -> Result<()> {
    let _guard = home_lock().lock().unwrap();
    let temp_home = TempDirGuard::new("home")?;
    let trained_dir = temp_home.path().join(".turbocalm").join("trained");
    let (config, checkpoint_path) = train_checkpoint(&trained_dir, 32)?;

    let config_path = temp_home.path().join("engine-config.json");
    std::fs::write(&config_path, serde_json::to_vec_pretty(&config)?)?;

    let home_guard = HomeEnvGuard::set(temp_home.path());
    let result = (|| -> Result<()> {
        let texts = vec![
            "steady breathing practice".to_string(),
            "steady breathing practice for focus".to_string(),
        ];
        let input_ids = encode_text_batch(&texts, &config, &Device::Cpu)?;

        let mut engine = EmbeddingEngine::new(&config_path);
        engine.load_model()?;

        let engine_pooled = engine.embed(&input_ids)?.mean(1)?;
        let expected_model =
            EmbeddingModel::from_checkpoint(&checkpoint_path, config.clone(), Device::Cpu)?;
        let expected_pooled = expected_model.encode_input_ids_pooled(&input_ids)?;

        let engine_rows = tensor_rows(&engine_pooled)?;
        let expected_rows = tensor_rows(&expected_pooled)?;
        assert_eq!(engine_rows.len(), expected_rows.len());
        assert!(
            max_abs_diff(&engine_rows, &expected_rows) < 1e-4,
            "expected EmbeddingEngine to load trained checkpoint output"
        );

        let similarity = cosine_similarity(&engine_rows[0], &engine_rows[1]);
        assert!(
            similarity.abs() > 1e-6,
            "expected non-zero cosine similarity for similar texts, got {similarity}"
        );
        Ok(())
    })();
    drop(home_guard);
    result
}

fn train_checkpoint(
    checkpoint_dir: &Path,
    latent_size: usize,
) -> Result<(CalmAutoencoderConfig, PathBuf)> {
    std::fs::create_dir_all(checkpoint_dir)?;

    let config = small_model_config(latent_size);
    let training_config = TrainingConfig {
        batch_size: 6,
        lr: 0.02,
        weight_decay: 0.0,
        temperature: 0.07,
        max_epochs: 6,
        eval_interval: 1,
        patience: 6,
        checkpoint_dir: checkpoint_dir.to_path_buf(),
        min_corpus_size: 1,
    };

    let mut trainer = Trainer::new(config.clone(), training_config, Device::Cpu)?;
    let summary = trainer.train(&synthetic_training_corpus())?;
    summary
        .best_checkpoint
        .as_ref()
        .context("expected a best checkpoint to be written")?;

    let latest_path = latest_checkpoint_path_in_dir(checkpoint_dir);
    assert!(latest_path.exists(), "expected {}", latest_path.display());
    Ok((config, latest_path))
}

fn small_model_config(latent_size: usize) -> CalmAutoencoderConfig {
    CalmAutoencoderConfig {
        vocab_size: 512,
        hidden_size: 32,
        intermediate_size: 64,
        latent_size,
        num_encoder_layers: 2,
        num_decoder_layers: 2,
        patch_size: 4,
        max_position_embeddings: 128,
        tie_word_embeddings: true,
        ..Default::default()
    }
}

fn synthetic_training_corpus() -> Corpus {
    let themes = [
        (
            "breathing",
            "steady breathing inhale exhale calm focus relaxation",
        ),
        (
            "architecture",
            "distributed systems event sourcing projection storage",
        ),
        (
            "journaling",
            "daily reflection gratitude notes planning review",
        ),
        ("debugging", "trace logs reproduction isolate failure fix"),
        (
            "nutrition",
            "balanced meals vegetables protein hydration energy",
        ),
    ];

    let mut pairs = Vec::new();
    for (theme, phrase) in themes {
        for idx in 0..6 {
            pairs.push(TrainingPair {
                anchor: format!("{theme} anchor {idx} {phrase}"),
                positive: format!("{theme} positive {idx} {phrase}"),
            });
        }
    }

    Corpus {
        metadata: CorpusMetadata {
            pair_count: pairs.len(),
            category_count: themes.len(),
            source_count: 1,
            categorized_pair_count: pairs.len(),
            temporal_pair_count: 0,
        },
        pairs,
    }
}

fn synthetic_eval_corpus() -> EvalCorpus {
    EvalCorpus {
        pairs: vec![
            EvalPair {
                query: "calm inhale exhale breathing focus".to_string(),
                relevant_ids: vec!["doc-breathing".to_string()],
            },
            EvalPair {
                query: "event sourcing system projections".to_string(),
                relevant_ids: vec!["doc-architecture".to_string()],
            },
            EvalPair {
                query: "trace reproduction debugging fix".to_string(),
                relevant_ids: vec!["doc-debugging".to_string()],
            },
        ],
        documents: vec![
            (
                "doc-breathing".to_string(),
                "steady breathing inhale exhale calm focus relaxation".to_string(),
            ),
            (
                "doc-architecture".to_string(),
                "distributed systems event sourcing projection storage".to_string(),
            ),
            (
                "doc-journaling".to_string(),
                "daily reflection gratitude notes planning review".to_string(),
            ),
            (
                "doc-debugging".to_string(),
                "trace logs reproduction isolate failure fix".to_string(),
            ),
            (
                "doc-nutrition".to_string(),
                "balanced meals vegetables protein hydration energy".to_string(),
            ),
            (
                "doc-distractor-1".to_string(),
                "astronomy telescope observation planetary motion".to_string(),
            ),
            (
                "doc-distractor-2".to_string(),
                "gardening compost seedlings watering sunlight".to_string(),
            ),
        ],
    }
}

fn tensor_rows(tensor: &candle_core::Tensor) -> Result<Vec<Vec<f32>>> {
    let (rows, dims) = tensor.dims2()?;
    let values = tensor
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    Ok(values
        .chunks(dims)
        .take(rows)
        .map(|chunk| chunk.to_vec())
        .collect())
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut left_norm = 0.0f32;
    let mut right_norm = 0.0f32;

    for (left_value, right_value) in left.iter().zip(right.iter()) {
        dot += left_value * right_value;
        left_norm += left_value * left_value;
        right_norm += right_value * right_value;
    }

    dot / (left_norm.sqrt() * right_norm.sqrt()).max(1e-6)
}

fn max_abs_diff(left: &[Vec<f32>], right: &[Vec<f32>]) -> f32 {
    left.iter()
        .flat_map(|row| row.iter())
        .zip(right.iter().flat_map(|row| row.iter()))
        .map(|(left_value, right_value)| (left_value - right_value).abs())
        .fold(0.0f32, f32::max)
}

fn home_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

struct HomeEnvGuard {
    original: Option<OsString>,
}

impl HomeEnvGuard {
    fn set(path: &Path) -> Self {
        let original = std::env::var_os("HOME");
        std::env::set_var("HOME", path);
        Self { original }
    }
}

impl Drop for HomeEnvGuard {
    fn drop(&mut self) {
        match &self.original {
            Some(path) => std::env::set_var("HOME", path),
            None => std::env::remove_var("HOME"),
        }
    }
}

struct TempDirGuard {
    path: PathBuf,
}

impl TempDirGuard {
    fn new(label: &str) -> Result<Self> {
        let unique = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let path = std::env::temp_dir().join(format!(
            "turbocalm-train-sprint3-{label}-{}-{unique}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path)?;
        Ok(Self { path })
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}
