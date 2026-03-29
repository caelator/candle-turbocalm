pub mod calibrate;
pub mod checkpoint;
pub mod corpus;
pub mod embedding;
pub mod eval;
pub mod loss;
pub mod online;
pub mod pairs;
pub mod server;
pub mod spike;
pub mod trainer;

pub use calibrate::{
    recommend_thresholds, run_calibration, save_calibration_toml, CalibrationCorpus,
    CalibrationPair, CalibrationReport, CalibrationStats, RecommendedThresholds, SimilarityTier,
};
pub use checkpoint::{
    checkpoint_path_for_version, default_checkpoint_dir, list_checkpoints, load_checkpoint,
    next_checkpoint_version, save_checkpoint, CheckpointInfo,
};
pub use corpus::{
    build_pairs_from_entries, dedup_entries, load_from_git_commits, load_from_jsonl,
    load_from_memory_lancedb, load_from_session_logs, merge_corpus_sources, save_to_jsonl,
    CorpusEntry,
};
pub use embedding::{EmbeddingMode, EmbeddingModel};
pub use eval::{run_eval, EvalCorpus, EvalMetrics, EvalPair};
pub use loss::{nt_xent_loss, DEFAULT_TEMPERATURE};
pub use online::{OnlineAddResult, OnlineLearner, OnlineLearningConfig};
pub use pairs::{Corpus as TrainingCorpus, CorpusMetadata, TrainingPair};
pub use server::{load_model_or_random, serve, serve_with_listener, DEFAULT_MODEL_NAME};
pub use trainer::{Trainer, TrainingConfig, TrainingSummary};
