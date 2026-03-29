//! Pareto front tracking for multi-objective optimization
//!
//! Maintains a set of non-dominated solutions throughout the calibration process.

use crate::{FitnessMetrics, QuantProfile};
use std::cmp::Ordering;

/// A solution with its associated fitness metrics
#[derive(Debug, Clone)]
pub struct ParetoSolution {
    pub profile: QuantProfile,
    pub fitness: FitnessMetrics,
    pub objective_value: f64, // Weighted combination for ranking
}

/// Pareto front tracker maintaining non-dominated solutions
#[derive(Debug, Clone)]
pub struct ParetoFront {
    solutions: Vec<ParetoSolution>,
    max_size: Option<usize>,
}

impl ParetoFront {
    /// Create new Pareto front tracker
    pub fn new(max_size: Option<usize>) -> Self {
        Self {
            solutions: Vec::new(),
            max_size,
        }
    }

    /// Add a new solution, updating the Pareto front
    pub fn add_solution(&mut self, solution: ParetoSolution) {
        // Check if new solution is dominated by any existing solution
        if self.is_dominated(&solution.fitness) {
            return; // Solution is dominated, don't add
        }

        // Remove all solutions dominated by the new solution
        self.solutions
            .retain(|existing| !dominates(&solution.fitness, &existing.fitness));

        // Add the new solution
        self.solutions.push(solution);

        // Apply size limit if specified
        if let Some(max_size) = self.max_size {
            if self.solutions.len() > max_size {
                self.trim_to_size(max_size);
            }
        }
    }

    /// Check if a fitness vector is dominated by any solution in the front
    pub fn is_dominated(&self, fitness: &FitnessMetrics) -> bool {
        self.solutions
            .iter()
            .any(|sol| dominates(&sol.fitness, fitness))
    }

    /// Get all non-dominated solutions
    pub fn get_solutions(&self) -> &[ParetoSolution] {
        &self.solutions
    }

    /// Get the best solution according to a specific criterion
    pub fn get_best_by_objective(&self) -> Option<&ParetoSolution> {
        self.solutions.iter().min_by(|a, b| {
            a.objective_value
                .partial_cmp(&b.objective_value)
                .unwrap_or(Ordering::Equal)
        })
    }

    /// Get solution with highest memory gain
    pub fn get_best_memory_gain(&self) -> Option<&ParetoSolution> {
        self.solutions.iter().max_by(|a, b| {
            a.fitness
                .memory_gain
                .partial_cmp(&b.fitness.memory_gain)
                .unwrap_or(Ordering::Equal)
        })
    }

    /// Get solution with lowest quality degradation
    pub fn get_best_quality(&self) -> Option<&ParetoSolution> {
        self.solutions.iter().min_by(|a, b| {
            let quality_a = a.fitness.delta_brier_lm + a.fitness.cosine_penalty;
            let quality_b = b.fitness.delta_brier_lm + b.fitness.cosine_penalty;
            quality_a.partial_cmp(&quality_b).unwrap_or(Ordering::Equal)
        })
    }

    /// Get number of solutions in the front
    pub fn size(&self) -> usize {
        self.solutions.len()
    }

    /// Check if the front is empty
    pub fn is_empty(&self) -> bool {
        self.solutions.is_empty()
    }

    /// Clear all solutions
    pub fn clear(&mut self) {
        self.solutions.clear();
    }

    /// Trim front to specified size using crowding distance
    fn trim_to_size(&mut self, target_size: usize) {
        if self.solutions.len() <= target_size {
            return;
        }

        // Calculate crowding distances
        let distances = self.calculate_crowding_distances();

        // Create pairs of (solution_index, distance) and sort by distance (descending)
        let mut indexed_distances: Vec<(usize, f64)> = distances.into_iter().enumerate().collect();
        indexed_distances.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        // Keep solutions with highest crowding distances
        let keep_indices: Vec<usize> = indexed_distances
            .iter()
            .take(target_size)
            .map(|(idx, _)| *idx)
            .collect();

        // Sort indices to maintain order
        let mut sorted_indices = keep_indices;
        sorted_indices.sort();

        // Create new solutions vector with selected solutions
        let mut new_solutions = Vec::with_capacity(target_size);
        for &idx in &sorted_indices {
            new_solutions.push(self.solutions[idx].clone());
        }

        self.solutions = new_solutions;
    }

    /// Calculate crowding distances for diversity preservation
    fn calculate_crowding_distances(&self) -> Vec<f64> {
        let n = self.solutions.len();
        let mut distances = vec![0.0; n];

        if n <= 2 {
            return vec![f64::INFINITY; n]; // Boundary solutions get infinite distance
        }

        // For each objective dimension
        for obj_idx in 0..4 {
            // 4 objectives: memory_gain, delta_brier_lm, cosine_penalty, latency_penalty
            // Create sorted indices for this objective
            let mut indices: Vec<usize> = (0..n).collect();
            indices.sort_by(|&a, &b| {
                let val_a = self.get_objective_value(&self.solutions[a].fitness, obj_idx);
                let val_b = self.get_objective_value(&self.solutions[b].fitness, obj_idx);
                val_a.partial_cmp(&val_b).unwrap_or(Ordering::Equal)
            });

            // Boundary points get infinite distance
            distances[indices[0]] = f64::INFINITY;
            distances[indices[n - 1]] = f64::INFINITY;

            // Calculate range for normalization
            let obj_min = self.get_objective_value(&self.solutions[indices[0]].fitness, obj_idx);
            let obj_max =
                self.get_objective_value(&self.solutions[indices[n - 1]].fitness, obj_idx);
            let range = obj_max - obj_min;

            if range > 0.0 {
                // Add crowding distance for intermediate points
                for i in 1..n - 1 {
                    let prev_val =
                        self.get_objective_value(&self.solutions[indices[i - 1]].fitness, obj_idx);
                    let next_val =
                        self.get_objective_value(&self.solutions[indices[i + 1]].fitness, obj_idx);
                    distances[indices[i]] += (next_val - prev_val) / range;
                }
            }
        }

        distances
    }

    /// Get objective value by index
    fn get_objective_value(&self, fitness: &FitnessMetrics, obj_idx: usize) -> f64 {
        match obj_idx {
            0 => fitness.memory_gain,
            1 => fitness.delta_brier_lm,
            2 => fitness.cosine_penalty,
            3 => fitness.latency_penalty,
            _ => panic!("Invalid objective index"),
        }
    }
}

/// Check if fitness vector `a` dominates fitness vector `b`
/// For our problem: maximize memory_gain, minimize everything else
pub fn dominates(a: &FitnessMetrics, b: &FitnessMetrics) -> bool {
    let better_or_equal = a.memory_gain >= b.memory_gain
        && a.delta_brier_lm <= b.delta_brier_lm
        && a.cosine_penalty <= b.cosine_penalty
        && a.latency_penalty <= b.latency_penalty;

    let strictly_better = a.memory_gain > b.memory_gain
        || a.delta_brier_lm < b.delta_brier_lm
        || a.cosine_penalty < b.cosine_penalty
        || a.latency_penalty < b.latency_penalty;

    better_or_equal && strictly_better
}

/// Non-dominated sorting for NSGA-II style algorithms
pub fn non_dominated_sort(solutions: &[ParetoSolution]) -> Vec<Vec<usize>> {
    let n = solutions.len();
    let mut fronts = Vec::new();
    let mut domination_counts = vec![0; n];
    let mut dominated_solutions: Vec<Vec<usize>> = vec![Vec::new(); n];

    // Calculate domination relationships
    for i in 0..n {
        for j in 0..n {
            if i != j {
                if dominates(&solutions[i].fitness, &solutions[j].fitness) {
                    dominated_solutions[i].push(j);
                } else if dominates(&solutions[j].fitness, &solutions[i].fitness) {
                    domination_counts[i] += 1;
                }
            }
        }
    }

    // Find first front (non-dominated solutions)
    let mut current_front = Vec::new();
    for i in 0..n {
        if domination_counts[i] == 0 {
            current_front.push(i);
        }
    }

    // Generate subsequent fronts
    while !current_front.is_empty() {
        fronts.push(current_front.clone());
        let mut next_front = Vec::new();

        for &solution_idx in &current_front {
            for &dominated_idx in &dominated_solutions[solution_idx] {
                domination_counts[dominated_idx] -= 1;
                if domination_counts[dominated_idx] == 0 {
                    next_front.push(dominated_idx);
                }
            }
        }

        current_front = next_front;
    }

    fronts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ContinuousParams, QuantProfile};

    fn create_test_solution(
        memory_gain: f64,
        delta_brier: f64,
        cosine_penalty: f64,
        latency_penalty: f64,
    ) -> ParetoSolution {
        ParetoSolution {
            profile: QuantProfile {
                bit_width: 4,
                qjl_dim: 32,
                rotation_seed: 42,
                qjl_threshold: 0.0001,
                scale_mode: "per_token".to_string(),
                clipping_percentile: 0.95,
                scale_multiplier: 1.0,
            },
            fitness: FitnessMetrics {
                memory_gain,
                delta_brier_lm: delta_brier,
                cosine_penalty,
                latency_penalty,
            },
            objective_value: 0.0,
        }
    }

    #[test]
    fn test_dominance() {
        let a = FitnessMetrics {
            memory_gain: 2.0,
            delta_brier_lm: 0.1,
            cosine_penalty: 0.05,
            latency_penalty: 0.02,
        };

        let b = FitnessMetrics {
            memory_gain: 1.8,      // worse
            delta_brier_lm: 0.12,  // worse
            cosine_penalty: 0.05,  // equal
            latency_penalty: 0.02, // equal
        };

        assert!(dominates(&a, &b));
        assert!(!dominates(&b, &a));
    }

    #[test]
    fn test_pareto_front_maintenance() {
        let mut front = ParetoFront::new(Some(5));

        // Add some solutions
        front.add_solution(create_test_solution(2.0, 0.1, 0.05, 0.02));
        front.add_solution(create_test_solution(1.8, 0.08, 0.04, 0.01)); // Different trade-offs
        front.add_solution(create_test_solution(1.5, 0.05, 0.02, 0.008)); // Lower on all penalties

        assert_eq!(front.size(), 3);

        // Add dominated solution
        front.add_solution(create_test_solution(1.0, 0.2, 0.1, 0.05)); // Clearly dominated
        assert_eq!(front.size(), 3); // Should not be added

        // Add dominating solution
        front.add_solution(create_test_solution(2.5, 0.05, 0.02, 0.01)); // Dominates some existing
        assert!(front.size() <= 3); // Some solutions should be removed
    }

    #[test]
    fn test_non_dominated_sorting() {
        let solutions = vec![
            create_test_solution(2.0, 0.1, 0.05, 0.02),  // Front 1
            create_test_solution(1.8, 0.08, 0.04, 0.01), // Front 1 (trade-off)
            create_test_solution(1.5, 0.12, 0.06, 0.03), // Front 2 (dominated by first)
            create_test_solution(1.0, 0.15, 0.08, 0.04), // Front 3
        ];

        let fronts = non_dominated_sort(&solutions);

        assert!(fronts.len() >= 2);
        assert!(fronts[0].len() >= 1); // At least one solution in first front
    }
}
