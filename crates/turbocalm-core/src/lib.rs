//! turbocalm-core - Core types and utilities for TurboCALM
//!
//! This crate provides configuration types, device selection, error handling,
//! tokenizer loading, HuggingFace hub utilities, and metrics for the TurboCALM project.

pub mod config;
pub mod device;
pub mod error;
pub mod hub;
pub mod metrics;
pub mod quant;
pub mod tokenizer;

pub use config::{AutoencoderConfig, CALMConfig, RopeScaling};
pub use device::{
    auto_device, cached_device, DeviceInfo, DevicePreference, DeviceSelector, DeviceType,
};
pub use error::{AnyhowResult, Result, TurboCALMError};
pub use hub::{CompleteModelDownload, DownloadManifest, DownloadUtils, HubClient};
pub use metrics::{
    MemoryReporter, MemorySource, MemoryUsage, MetricsBundle, MetricsTracker, SimilarityMetrics,
    TensorMemoryInfo,
};
pub use quant::{ContinuousParams, QuantProfile};
pub use tokenizer::{SpecialTokens, TokenizerLoader, TokenizerType, TokenizerUtils};
