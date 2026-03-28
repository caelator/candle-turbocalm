//! turbocalm-checkpoint - Checkpoint download and conversion for TurboCALM
//!
//! This crate provides functionality for downloading HuggingFace checkpoints,
//! parsing PyTorch state dicts, tensor name remapping, shape verification,
//! and converting models to safetensors format.

pub mod convert;
pub mod download;
pub mod manifest;
pub mod parser;
pub mod remapping;
pub mod verification;

pub use convert::{ConvertArgs, ConvertCommand, ConvertHandler, run_convert_command};
pub use download::{CALMCheckpoint, CheckpointDownloader, KnownCALMModels};
pub use manifest::{CALMModelManifest, ManifestManager, ModelSummary};
pub use parser::{StateDictParser, TensorSummary};
pub use remapping::{
    RemappingPresets, RemappingUtils, TensorNameRemapper, TensorPatternAnalysis,
};
pub use verification::{ShapeVerifier, VerificationReport, VerificationStatus};
