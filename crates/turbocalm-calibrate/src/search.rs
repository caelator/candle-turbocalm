//! Search orchestrator combining discrete and continuous optimization
//!
//! Implements the two-level search strategy:
//! 1. Outer loop: enumerate discrete parameter combinations
//! 2. Inner loop: CMA-ES optimization of continuous parameters

use crate::{
    CalibrationConfig, ContinuousParams, FitnessMetrics, QuantProfile,
    cmaes::CmaEs,
    dataset::ProcessedDataset,
    objective::{BatchEvaluator, ObjectiveFunction, ReferenceMetrics, create_reference_metrics},
    pareto::{ParetoFront, ParetoSolution},
};
use anyhow::Result;
use candle_core::Device;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;
use tracing::{info, warn, debug};

/// Search orchestrator for calibration optimization
pub struct CalibrationSearch {
    config: CalibrationConfig,
    device: Device,
    evaluator: BatchEvaluator,
    pareto_front: ParetoFront,
    iteration_count: usize,
    best_solutions: HashMap<String, ParetoSolution>,
}

/// Search progress information
#[derive(Debug, Clone)]
pub struct SearchProgress {
    /// Current iteration
    pub iteration: usize,
    /// Total iterations completed
    pub total_iterations: usize,
    /// Current discrete configuration being explored
    pub current_discrete_config: String,
    /// Number of solutions in Pareto front
    pub pareto_size: usize,
    /// Best objective value found so far
    pub best_objective: f64,
    /// Best fitness metrics
    pub best_fitness: FitnessMetrics,
}

/// Results from calibration search
#[derive(Debug, Clone)]
pub struct SearchResults {
    /// All non-dominated solutions
    pub pareto_solutions: Vec<ParetoSolution>,
    /// Best solution by objective value
    pub best_solution: ParetoSolution,
    /// Search statistics
    pub statistics: SearchStatistics,
}

/// Search performance statistics
#[derive(Debug, Clone)]
pub struct SearchStatistics {
    /// Total iterations performed
    pub total_iterations: usize,
    /// Number of discrete configurations explored
    pub discrete_configs_explored: usize,
    /// Average CMA-ES iterations per discrete config
    pub avg_cmaes_iterations: f64,
    /// Total evaluation time (ms)
    pub total_time_ms: f64,
    /// Evaluations per second
    pub evaluations_per_second: f64,
}

impl CalibrationSearch {
    /// Create new calibration search
    pub fn new(
        config: CalibrationConfig,
        device: Device,
    ) -> Result<Self> {
        let evaluator = BatchEvaluator::new(config.weights.clone());
        let pareto_front = ParetoFront::new(Some(100)); // Limit Pareto front size

        Ok(Self {
            config,
            device,
            evaluator,
            pareto_front,
            iteration_count: 0,
            best_solutions: HashMap::new(),
        })
    }

    /// Run complete calibration search
    pub fn run_search(
        &mut self,
        dataset: &ProcessedDataset,
        progress_callback: Option<Box<dyn Fn(&SearchProgress)>>,
    ) -> Result<SearchResults> {
        info!("Starting calibration search with {} discrete configurations",
              self.count_discrete_configs());

        let start_time = std::time::Instant::now();

        // Establish reference metrics
        let reference_metrics = create_reference_metrics(dataset, &self.device)?;
        self.evaluator.set_reference(reference_metrics.clone());

        let mut total_evaluations = 0;
        let mut discrete_configs_explored = 0;
        let mut cmaes_iterations_sum = 0;

        // Clone configuration vectors to avoid borrowing issues
        let bit_widths = self.config.discrete.bit_widths.clone();
        let qjl_dims = self.config.discrete.qjl_dims.clone();
        let rotation_seeds = self.config.discrete.rotation_seeds.clone();

        // Outer loop: enumerate discrete configurations
        for bit_width in &bit_widths {
            for qjl_dim in &qjl_dims {
                for rotation_seed in &rotation_seeds {
                    if self.iteration_count >= self.config.max_total_iterations {
                        warn!("Reached maximum total iterations, stopping search");
                        break;
                    }

                    let discrete_config = format!("{}bit_{}qjl_{}seed",
                                                  bit_width, qjl_dim, rotation_seed);

                    info!("Exploring discrete config: {}", discrete_config);

                    // Inner loop: CMA-ES optimization for this discrete configuration
                    let (best_profile, cmaes_iters) = self.optimize_continuous_parameters(
                        *bit_width,
                        *qjl_dim,
                        *rotation_seed,
                        dataset,
                        &progress_callback,
                    )?;

                    total_evaluations += cmaes_iters * self.config.cmaes_population_size;
                    discrete_configs_explored += 1;
                    cmaes_iterations_sum += cmaes_iters;

                    // Store best solution for this discrete config
                    self.best_solutions.insert(discrete_config, best_profile);

                    // Report progress
                    if let Some(ref callback) = progress_callback {
                        let progress = SearchProgress {
                            iteration: self.iteration_count,
                            total_iterations: total_evaluations,
                            current_discrete_config: format!("{}bit_{}qjl_{}seed",
                                                             bit_width, qjl_dim, rotation_seed),
                            pareto_size: self.pareto_front.size(),
                            best_objective: self.pareto_front.get_best_by_objective()
                                .map_or(f64::INFINITY, |s| s.objective_value),
                            best_fitness: self.pareto_front.get_best_by_objective()
                                .map_or(
                                    FitnessMetrics {
                                        memory_gain: 0.0,
                                        delta_brier_lm: f64::INFINITY,
                                        cosine_penalty: 1.0,
                                        latency_penalty: 1.0
                                    },
                                    |s| s.fitness.clone()
                                ),
                        };
                        callback(&progress);
                    }
                }
            }
        }

        let total_time = start_time.elapsed();

        // Generate final results
        let pareto_solutions = self.pareto_front.get_solutions().to_vec();
        let best_solution = self.pareto_front.get_best_by_objective()
            .ok_or_else(|| anyhow::anyhow!("No solutions found"))?
            .clone();

        let statistics = SearchStatistics {
            total_iterations: total_evaluations,
            discrete_configs_explored,
            avg_cmaes_iterations: if discrete_configs_explored > 0 {
                cmaes_iterations_sum as f64 / discrete_configs_explored as f64
            } else {
                0.0
            },
            total_time_ms: total_time.as_millis() as f64,
            evaluations_per_second: total_evaluations as f64 / total_time.as_secs_f64(),
        };

        info!("Search completed: {} solutions in Pareto front, {} total evaluations in {:.2}s",
              pareto_solutions.len(), total_evaluations, total_time.as_secs_f64());

        Ok(SearchResults {
            pareto_solutions,
            best_solution,
            statistics,
        })
    }

    /// Optimize continuous parameters for a given discrete configuration
    fn optimize_continuous_parameters(
        &mut self,
        bit_width: u8,
        qjl_dim: usize,
        rotation_seed: u64,
        dataset: &ProcessedDataset,
        progress_callback: &Option<Box<dyn Fn(&SearchProgress)>>,
    ) -> Result<(ParetoSolution, usize)> {
        debug!("Starting CMA-ES for bit_width={}, qjl_dim={}, rotation_seed={}",
               bit_width, qjl_dim, rotation_seed);

        // Initialize CMA-ES with default continuous parameters
        let initial_continuous = ContinuousParams::default();
        let mut cmaes = CmaEs::new(
            &initial_continuous,
            self.config.cmaes_population_size,
            Some(rotation_seed + 12345), // Ensure different seed for CMA-ES
        );

        let mut best_solution: Option<ParetoSolution> = None;
        let mut cmaes_iteration = 0;

        for iteration in 0..self.config.max_cmaes_iterations {
            if self.iteration_count >= self.config.max_total_iterations {
                break;
            }

            // Generate population
            let continuous_population = cmaes.ask();

            // Create full profiles for evaluation
            let profiles: Vec<QuantProfile> = continuous_population.iter()
                .map(|continuous| QuantProfile {
                    bit_width,
                    qjl_dim,
                    rotation_seed,
                    continuous: continuous.clone(),
                })
                .collect();

            // Evaluate population
            let evaluations = self.evaluator.evaluate_batch(&profiles, dataset, &self.device)?;

            // Extract fitness values for CMA-ES (minimize objective)
            let fitness_values: Vec<f64> = evaluations.iter()
                .map(|(_, objective)| *objective)
                .collect();

            // Update CMA-ES
            cmaes.tell(&continuous_population, &fitness_values)?;

            // Update Pareto front with all solutions
            for (profile, (fitness, objective)) in profiles.iter().zip(evaluations.iter()) {
                let solution = ParetoSolution {
                    profile: profile.clone(),
                    fitness: fitness.clone(),
                    objective_value: *objective,
                };

                self.pareto_front.add_solution(solution.clone());

                // Track best solution for this discrete config
                if best_solution.as_ref().map_or(true, |best| objective < &best.objective_value) {
                    best_solution = Some(solution);
                }
            }

            self.iteration_count += self.config.cmaes_population_size;
            cmaes_iteration = iteration + 1;

            // Check convergence
            if cmaes.has_converged() {
                debug!("CMA-ES converged after {} iterations", cmaes_iteration);
                break;
            }
        }

        let final_solution = best_solution
            .ok_or_else(|| anyhow::anyhow!("No valid solutions found in CMA-ES"))?;

        debug!("CMA-ES completed: {} iterations, best objective = {:.6}",
               cmaes_iteration, final_solution.objective_value);

        Ok((final_solution, cmaes_iteration))
    }

    /// Count total discrete configurations
    fn count_discrete_configs(&self) -> usize {
        self.config.discrete.bit_widths.len() *
        self.config.discrete.qjl_dims.len() *
        self.config.discrete.rotation_seeds.len()
    }

    /// Get current Pareto front
    pub fn get_pareto_front(&self) -> &ParetoFront {
        &self.pareto_front
    }

    /// Get best solutions by discrete configuration
    pub fn get_best_solutions(&self) -> &HashMap<String, ParetoSolution> {
        &self.best_solutions
    }
}

/// Utility for resuming search from previous results
pub struct SearchResume {
    previous_solutions: Vec<ParetoSolution>,
}

impl SearchResume {
    /// Create resume state from previous results
    pub fn from_results(results: &SearchResults) -> Self {
        Self {
            previous_solutions: results.pareto_solutions.clone(),
        }
    }

    /// Initialize search with previous solutions
    pub fn initialize_search(&self, search: &mut CalibrationSearch) {
        for solution in &self.previous_solutions {
            search.pareto_front.add_solution(solution.clone());
        }

        info!("Resumed search with {} previous solutions", self.previous_solutions.len());
    }
}

/// Factory for creating different search strategies
pub struct SearchFactory;

impl SearchFactory {
    /// Create standard exhaustive search
    pub fn create_exhaustive_search(
        device: Device,
        max_iterations: Option<usize>,
    ) -> Result<CalibrationSearch> {
        let mut config = CalibrationConfig::default();
        if let Some(max_iter) = max_iterations {
            config.max_total_iterations = max_iter;
        }

        CalibrationSearch::new(config, device)
    }

    /// Create focused search with limited parameter space
    pub fn create_focused_search(
        device: Device,
        target_bit_width: Option<u8>,
        target_qjl_dim: Option<usize>,
    ) -> Result<CalibrationSearch> {
        let mut config = CalibrationConfig::default();

        if let Some(bits) = target_bit_width {
            config.discrete.bit_widths = vec![bits];
        }

        if let Some(dim) = target_qjl_dim {
            config.discrete.qjl_dims = vec![dim];
        }

        // Reduce iterations for focused search
        config.max_cmaes_iterations = 30;
        config.max_total_iterations = 500;

        CalibrationSearch::new(config, device)
    }

    /// Create rapid prototyping search with minimal iterations
    pub fn create_rapid_search(device: Device) -> Result<CalibrationSearch> {
        let mut config = CalibrationConfig::default();
        config.discrete.bit_widths = vec![4]; // Only 4-bit for speed
        config.discrete.qjl_dims = vec![32]; // Single QJL dimension
        config.discrete.rotation_seeds = vec![42]; // Single seed
        config.max_cmaes_iterations = 10;
        config.cmaes_population_size = 6;
        config.max_total_iterations = 100;

        CalibrationSearch::new(config, device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{CalibrationDataset, CalibrationSample};

    fn create_test_dataset() -> Result<ProcessedDataset> {
        let samples = vec![
            CalibrationSample {
                text: "Hello world".to_string(),
                metadata: None,
            },
            CalibrationSample {
                text: "This is a test".to_string(),
                metadata: None,
            },
        ];

        let dataset = CalibrationDataset { samples };

        // For testing, create a minimal processed dataset
        // In practice, this would use a real tokenizer
        use candle_core::{Device, Tensor};
        let device = Device::Cpu;

        Ok(ProcessedDataset {
            input_ids: vec![vec![1, 2, 3], vec![4, 5, 6]],
            attention_masks: vec![vec![1, 1, 1], vec![1, 1, 1]],
            kv_traces: vec![],
            device: device.clone(),
        })
    }

    #[test]
    fn test_search_factory() -> Result<()> {
        let device = Device::Cpu;

        let _exhaustive = SearchFactory::create_exhaustive_search(device.clone(), Some(100))?;
        let _focused = SearchFactory::create_focused_search(device.clone(), Some(4), Some(32))?;
        let _rapid = SearchFactory::create_rapid_search(device)?;

        Ok(())
    }

    #[test]
    fn test_discrete_config_count() -> Result<()> {
        let device = Device::Cpu;
        let search = SearchFactory::create_exhaustive_search(device, Some(100))?;

        let count = search.count_discrete_configs();
        assert!(count > 0);

        Ok(())
    }
}