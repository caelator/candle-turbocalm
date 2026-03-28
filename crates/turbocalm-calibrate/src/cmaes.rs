//! CMA-ES (Covariance Matrix Adaptation Evolution Strategy) implementation
//!
//! Self-contained implementation optimized for continuous parameter optimization
//! in the turbocalm calibration pipeline.

use crate::ContinuousParams;
use anyhow::Result;
use rand::{Rng, SeedableRng};
use rand_distr::Normal;

/// CMA-ES optimizer for continuous parameters
#[derive(Debug, Clone)]
pub struct CmaEs {
    /// Dimension of the parameter space
    dimension: usize,
    /// Population size (lambda)
    lambda: usize,
    /// Parent population size (mu)
    mu: usize,
    /// Current mean vector
    mean: Vec<f64>,
    /// Step size (sigma)
    sigma: f64,
    /// Covariance matrix
    covariance: Vec<Vec<f64>>,
    /// Evolution path for covariance
    pc: Vec<f64>,
    /// Evolution path for sigma
    ps: Vec<f64>,
    /// Weights for recombination
    weights: Vec<f64>,
    /// Effective variance of the weights
    mu_eff: f64,
    /// Learning rates and strategy parameters
    cc: f64,
    cs: f64,
    c1: f64,
    cmu: f64,
    damps: f64,
    /// Expected norm of N(0,I)
    chi_n: f64,
    /// Random number generator
    rng: rand::rngs::StdRng,
    /// Generation counter
    generation: usize,
}

impl CmaEs {
    /// Create new CMA-ES optimizer
    pub fn new(
        initial_params: &ContinuousParams,
        population_size: usize,
        seed: Option<u64>,
    ) -> Self {
        let dimension = 3; // clipping_percentile, scale_multiplier, qjl_threshold
        let lambda = population_size;
        let mu = (lambda / 2).max(1);

        // Initialize mean from parameters
        let mean = vec![
            initial_params.clipping_percentile,
            initial_params.scale_multiplier,
            initial_params.qjl_threshold,
        ];

        // Strategy parameters (Hansen & Ostermeier, 2001)
        let weights = Self::recombination_weights(mu);
        let mu_eff = 1.0 / weights.iter().map(|w| w.powi(2)).sum::<f64>();

        let cc = (4.0 + mu_eff / dimension as f64)
            / (dimension as f64 + 4.0 + 2.0 * mu_eff / dimension as f64);
        let cs = (mu_eff + 2.0) / (dimension as f64 + mu_eff + 5.0);
        let c1 = 2.0 / ((dimension as f64 + 1.3).powi(2) + mu_eff);
        let cmu = 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((dimension as f64 + 2.0).powi(2) + mu_eff);
        let damps =
            1.0 + 2.0 * (0.0_f64.max((mu_eff - 1.0) / (dimension as f64 + 1.0) - 1.0)).sqrt() + cs;

        // Expected norm of N(0,I)
        let chi_n = (dimension as f64).sqrt()
            * (1.0 - 1.0 / (4.0 * dimension as f64)
                + 1.0 / (21.0 * dimension as f64 * dimension as f64));

        // Initialize covariance as identity matrix
        let mut covariance = vec![vec![0.0; dimension]; dimension];
        for i in 0..dimension {
            covariance[i][i] = 1.0;
        }

        let rng = match seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None => rand::rngs::StdRng::from_entropy(),
        };

        Self {
            dimension,
            lambda,
            mu,
            mean,
            sigma: 0.3, // Initial step size
            covariance,
            pc: vec![0.0; dimension],
            ps: vec![0.0; dimension],
            weights,
            mu_eff,
            cc,
            cs,
            c1,
            cmu,
            damps,
            chi_n,
            rng,
            generation: 0,
        }
    }

    /// Generate population for evaluation
    pub fn ask(&mut self) -> Vec<ContinuousParams> {
        let mut population = Vec::with_capacity(self.lambda);

        for _ in 0..self.lambda {
            // Sample from multivariate normal distribution N(mean, sigma^2 * C)
            let z = self.sample_standard_normal();
            let y = self.matrix_multiply_vector(&self.covariance_sqrt(), &z);

            let mut individual = Vec::with_capacity(self.dimension);
            for i in 0..self.dimension {
                individual.push(self.mean[i] + self.sigma * y[i]);
            }

            // Convert back to ContinuousParams with proper bounds
            population.push(ContinuousParams {
                clipping_percentile: individual[0].max(0.01).min(0.99),
                scale_multiplier: individual[1].max(0.1).min(10.0),
                qjl_threshold: individual[2].max(1e-6).min(1e-2),
            });
        }

        population
    }

    /// Update distribution based on fitness evaluations
    pub fn tell(&mut self, population: &[ContinuousParams], fitness: &[f64]) -> Result<()> {
        if population.len() != self.lambda || fitness.len() != self.lambda {
            anyhow::bail!("Population size mismatch");
        }

        // Sort by fitness (minimize)
        let mut indices: Vec<usize> = (0..self.lambda).collect();
        indices.sort_by(|&a, &b| fitness[a].partial_cmp(&fitness[b]).unwrap());

        // Convert parameters back to vectors
        let param_vectors: Vec<Vec<f64>> = population
            .iter()
            .map(|p| vec![p.clipping_percentile, p.scale_multiplier, p.qjl_threshold])
            .collect();

        // Recombination: update mean
        let old_mean = self.mean.clone();
        for i in 0..self.dimension {
            self.mean[i] = 0.0;
            for j in 0..self.mu {
                self.mean[i] += self.weights[j] * param_vectors[indices[j]][i];
            }
        }

        // Update evolution paths
        let mean_diff: Vec<f64> = (0..self.dimension)
            .map(|i| (self.mean[i] - old_mean[i]) / self.sigma)
            .collect();

        // Update ps (evolution path for sigma)
        let c_sigma = (self.cs * (2.0 - self.cs) * self.mu_eff).sqrt();
        for i in 0..self.dimension {
            self.ps[i] = (1.0 - self.cs) * self.ps[i] + c_sigma * mean_diff[i];
        }

        // Update pc (evolution path for covariance)
        let h_sig = if self.vector_norm(&self.ps)
            / (1.0 - (1.0 - self.cs).powi(2 * (self.generation + 1) as i32)).sqrt()
            < 1.4 + 2.0 / (self.dimension as f64 + 1.0)
        {
            1.0
        } else {
            0.0
        };

        let c_cov = (self.cc * (2.0 - self.cc) * self.mu_eff).sqrt();
        for i in 0..self.dimension {
            self.pc[i] = (1.0 - self.cc) * self.pc[i] + h_sig * c_cov * mean_diff[i];
        }

        // Update covariance matrix
        let artmp: Vec<Vec<f64>> = (0..self.mu)
            .map(|j| {
                (0..self.dimension)
                    .map(|i| (param_vectors[indices[j]][i] - old_mean[i]) / self.sigma)
                    .collect()
            })
            .collect();

        for i in 0..self.dimension {
            for j in 0..=i {
                let mut cov_update = (1.0 - self.c1 - self.cmu) * self.covariance[i][j];

                // Rank-one update
                cov_update += self.c1 * self.pc[i] * self.pc[j];

                // Rank-mu update
                for k in 0..self.mu {
                    cov_update += self.cmu * self.weights[k] * artmp[k][i] * artmp[k][j];
                }

                self.covariance[i][j] = cov_update;
                if i != j {
                    self.covariance[j][i] = cov_update;
                }
            }
        }

        // Update step size (sigma)
        let ps_norm = self.vector_norm(&self.ps);
        self.sigma *= (self.cs / self.damps * (ps_norm / self.chi_n - 1.0)).exp();

        self.generation += 1;
        Ok(())
    }

    /// Get current best estimate
    pub fn current_best(&self) -> ContinuousParams {
        ContinuousParams {
            clipping_percentile: self.mean[0].max(0.01).min(0.99),
            scale_multiplier: self.mean[1].max(0.1).min(10.0),
            qjl_threshold: self.mean[2].max(1e-6).min(1e-2),
        }
    }

    /// Check convergence criteria
    pub fn has_converged(&self) -> bool {
        // Simple convergence check based on step size
        self.sigma < 1e-12 || self.generation > 1000
    }

    fn recombination_weights(mu: usize) -> Vec<f64> {
        let raw_weights: Vec<f64> = (1..=mu)
            .map(|rank| (mu as f64 + 0.5).ln() - (rank as f64).ln())
            .collect();

        // Verify all weights are positive (should always be true for rank 1..=mu)
        // Since ln(mu+0.5) > ln(mu) >= ln(rank) for all rank in 1..=mu
        debug_assert!(
            raw_weights.iter().all(|&w| w > 0.0),
            "All recombination weights should be positive for ranks 1..=mu. mu={}, weights={:?}",
            mu,
            raw_weights
        );

        let sum_weights = raw_weights.iter().sum::<f64>();
        debug_assert!(sum_weights > 0.0, "Sum of weights should be positive");

        raw_weights
            .into_iter()
            .map(|weight| weight / sum_weights)
            .collect()
    }

    // Helper methods
    fn sample_standard_normal(&mut self) -> Vec<f64> {
        let normal = Normal::new(0.0, 1.0).unwrap();
        (0..self.dimension)
            .map(|_| self.rng.sample(normal))
            .collect()
    }

    fn vector_norm(&self, v: &[f64]) -> f64 {
        v.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    fn matrix_multiply_vector(&self, matrix: &[Vec<f64>], vector: &[f64]) -> Vec<f64> {
        matrix
            .iter()
            .map(|row| row.iter().zip(vector.iter()).map(|(a, b)| a * b).sum())
            .collect()
    }

    fn covariance_sqrt(&self) -> Vec<Vec<f64>> {
        // Proper Cholesky decomposition for 3x3 covariance matrix
        // Given positive-definite symmetric matrix C, compute L such that C = L * L^T
        let mut l = vec![vec![0.0; self.dimension]; self.dimension];

        for i in 0..self.dimension {
            for j in 0..=i {
                if i == j {
                    // Diagonal element: L[i][i] = sqrt(C[i][i] - sum(L[i][k]^2 for k < i))
                    let mut sum = 0.0;
                    for k in 0..j {
                        sum += l[i][k] * l[i][k];
                    }
                    l[i][j] = (self.covariance[i][i] - sum).sqrt();
                } else {
                    // Off-diagonal element: L[i][j] = (C[i][j] - sum(L[i][k] * L[j][k] for k < j)) / L[j][j]
                    let mut sum = 0.0;
                    for k in 0..j {
                        sum += l[i][k] * l[j][k];
                    }
                    l[i][j] = (self.covariance[i][j] - sum) / l[j][j];
                }
            }
        }

        l
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected:.12}, got {actual:.12}",
        );
    }

    #[test]
    fn test_cmaes_recombination_weights_follow_standard_log_formula() {
        let initial = ContinuousParams::default();
        let cmaes = CmaEs::new(&initial, 10, Some(42));

        let raw_weights: Vec<f64> = (1..=cmaes.mu)
            .map(|rank| (cmaes.mu as f64 + 0.5).ln() - (rank as f64).ln())
            .collect();
        let sum_raw = raw_weights.iter().sum::<f64>();
        let expected_weights: Vec<f64> =
            raw_weights.iter().map(|weight| weight / sum_raw).collect();

        assert_eq!(cmaes.weights.len(), cmaes.mu);
        assert!(cmaes.weights.iter().all(|weight| *weight > 0.0));
        assert_close(cmaes.weights.iter().sum::<f64>(), 1.0, 1e-12);

        for (actual, expected) in cmaes.weights.iter().zip(expected_weights.iter()) {
            assert_close(*actual, *expected, 1e-12);
        }

        let expected_mu_eff = 1.0
            / expected_weights
                .iter()
                .map(|weight| weight.powi(2))
                .sum::<f64>();
        assert_close(cmaes.mu_eff, expected_mu_eff, 1e-12);
    }

    #[test]
    fn test_cmaes_basic_functionality() {
        // Test basic CMA-ES functionality without requiring true convergence
        let initial = ContinuousParams::default();
        let mut cmaes = CmaEs::new(&initial, 10, Some(42));

        // Test population generation
        let population1 = cmaes.ask();
        assert_eq!(population1.len(), 10);

        // Test that all parameters are within bounds
        for params in &population1 {
            assert!(params.clipping_percentile >= 0.01 && params.clipping_percentile <= 0.99);
            assert!(params.scale_multiplier >= 0.1 && params.scale_multiplier <= 10.0);
            assert!(params.qjl_threshold >= 1e-6 && params.qjl_threshold <= 1e-2);
        }

        // Test fitness evaluation and tell
        let fitness: Vec<f64> = population1
            .iter()
            .map(|params| {
                // Simple fitness function - just return a random-like value based on params
                params.clipping_percentile + params.scale_multiplier + params.qjl_threshold * 1e4
            })
            .collect();

        // This should not panic
        cmaes.tell(&population1, &fitness).unwrap();

        // Test that we can generate a second population
        let population2 = cmaes.ask();
        assert_eq!(population2.len(), 10);

        // Test current_best returns valid parameters
        let best = cmaes.current_best();
        assert!(best.clipping_percentile >= 0.01 && best.clipping_percentile <= 0.99);
        assert!(best.scale_multiplier >= 0.1 && best.scale_multiplier <= 10.0);
        assert!(best.qjl_threshold >= 1e-6 && best.qjl_threshold <= 1e-2);
    }
}
