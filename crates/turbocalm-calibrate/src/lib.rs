//! TurboCalm Calibration Engine
//!
//! Evolutionary calibration engine for TurboQuant parameters using multi-objective optimization.
//! Combines discrete parameter enumeration with continuous optimization via CMA-ES.

use anyhow::Result;
use serde::{Deserialize, Serialize};
pub use turbocalm_core::{ContinuousParams, QuantProfile};

pub mod cmaes;
pub mod dataset;
pub mod objective;
pub mod pareto;
pub mod profiles;
pub mod report;
pub mod search;

/// Discrete parameter configuration space
#[derive(Debug, Clone)]
pub struct DiscreteConfig {
    pub bit_widths: Vec<u8>,
    pub qjl_dims: Vec<usize>,
    pub rotation_seeds: Vec<u64>,
}

impl Default for DiscreteConfig {
    fn default() -> Self {
        Self {
            bit_widths: vec![2, 3, 4],
            qjl_dims: vec![16, 32, 64],
            rotation_seeds: vec![42, 137, 256, 512],
        }
    }
}

/// Multi-objective fitness metrics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FitnessMetrics {
    /// Memory compression ratio gain
    pub memory_gain: f64,
    /// Change in Brier score for language modeling
    pub delta_brier_lm: f64,
    /// Cosine similarity penalty
    pub cosine_penalty: f64,
    /// Latency penalty
    pub latency_penalty: f64,
}

/// Objective function weights
#[derive(Debug, Clone)]
pub struct ObjectiveWeights {
    pub lambda1: f64, // Brier LM weight
    pub lambda2: f64, // Cosine penalty weight
    pub lambda3: f64, // Latency penalty weight
}

impl Default for ObjectiveWeights {
    fn default() -> Self {
        Self {
            lambda1: 1.0,
            lambda2: 0.5,
            lambda3: 0.3,
        }
    }
}

/// Calibration configuration
#[derive(Debug, Clone)]
pub struct CalibrationConfig {
    /// Discrete parameter space
    pub discrete: DiscreteConfig,
    /// Objective function weights
    pub weights: ObjectiveWeights,
    /// Maximum CMA-ES iterations per discrete config
    pub max_cmaes_iterations: usize,
    /// CMA-ES population size
    pub cmaes_population_size: usize,
    /// Overall iteration budget
    pub max_total_iterations: usize,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            discrete: DiscreteConfig::default(),
            weights: ObjectiveWeights::default(),
            max_cmaes_iterations: 50,
            cmaes_population_size: 10,
            max_total_iterations: 1000,
        }
    }
}
