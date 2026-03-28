//! Export and import of evolved QuantProfile configurations
//!
//! Handles serialization of calibration results to safetensors format
//! for integration with the turbocalm quantization pipeline.

use crate::search::SearchResults;
use crate::{FitnessMetrics, QuantProfile};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

/// Collection of evolved quantization profiles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileCollection {
    /// Metadata about the calibration run
    pub metadata: CalibrationMetadata,
    /// Pareto-optimal profiles
    pub profiles: Vec<ProfileEntry>,
    /// Best profile by overall objective
    pub best_profile: ProfileEntry,
}

/// Metadata for calibration run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationMetadata {
    /// Timestamp of calibration
    pub timestamp: String,
    /// Calibration configuration used
    pub config_summary: String,
    /// Dataset information
    pub dataset_info: DatasetInfo,
    /// Search statistics
    pub search_stats: SearchStats,
}

/// Dataset information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetInfo {
    /// Number of calibration samples
    pub num_samples: usize,
    /// Average sequence length
    pub avg_seq_length: f64,
    /// Dataset source/name
    pub source: Option<String>,
}

/// Search statistics summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchStats {
    /// Total evaluations performed
    pub total_evaluations: usize,
    /// Number of Pareto-optimal solutions found
    pub pareto_solutions: usize,
    /// Total search time in seconds
    pub search_time_seconds: f64,
}

/// Individual profile entry with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileEntry {
    /// The quantization profile
    pub profile: QuantProfile,
    /// Fitness metrics achieved
    pub fitness: FitnessMetrics,
    /// Weighted objective value
    pub objective_value: f64,
    /// Rank in Pareto front (0 = best)
    pub pareto_rank: usize,
    /// Profile identifier
    pub profile_id: String,
}

/// Profile export/import interface
pub struct ProfileExporter {
    /// Output directory for profiles
    output_dir: std::path::PathBuf,
}

impl ProfileExporter {
    /// Create new profile exporter
    pub fn new<P: AsRef<Path>>(output_dir: P) -> Result<Self> {
        let dir = output_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("Failed to create output directory: {:?}", dir))?;

        Ok(Self { output_dir: dir })
    }

    /// Export search results to safetensors format
    pub fn export_results(
        &self,
        results: &SearchResults,
        dataset_info: DatasetInfo,
        config_summary: String,
    ) -> Result<std::path::PathBuf> {
        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S").to_string();

        // Create profile collection
        let collection = self.create_profile_collection(
            results,
            dataset_info,
            config_summary,
            timestamp.clone(),
        )?;

        // Save to JSON (human readable)
        let json_path = self.output_dir.join(format!("profiles_{}.json", timestamp));
        self.save_as_json(&collection, &json_path)?;

        // Save to additional JSON format (for compatibility)
        let compact_json_path = self
            .output_dir
            .join(format!("profiles_{}_compact.json", timestamp));
        self.save_as_compact_json(&collection, &compact_json_path)?;

        // Save individual best profile for easy access
        let best_profile_path = self.output_dir.join("best_profile.json");
        self.save_best_profile(&collection.best_profile, &best_profile_path)?;

        tracing::info!(
            "Exported {} profiles to {:?}",
            collection.profiles.len(),
            compact_json_path
        );

        Ok(compact_json_path)
    }

    /// Import profiles from safetensors
    pub fn import_profiles<P: AsRef<Path>>(path: P) -> Result<ProfileCollection> {
        let path = path.as_ref();

        if path.extension().and_then(|s| s.to_str()) == Some("json") {
            Self::load_from_json(path)
        } else {
            Self::load_from_safetensors(path)
        }
    }

    /// Create profile collection from search results
    fn create_profile_collection(
        &self,
        results: &SearchResults,
        dataset_info: DatasetInfo,
        config_summary: String,
        timestamp: String,
    ) -> Result<ProfileCollection> {
        let metadata = CalibrationMetadata {
            timestamp,
            config_summary,
            dataset_info,
            search_stats: SearchStats {
                total_evaluations: results.statistics.total_iterations,
                pareto_solutions: results.pareto_solutions.len(),
                search_time_seconds: results.statistics.total_time_ms / 1000.0,
            },
        };

        // Convert Pareto solutions to profile entries
        let mut profiles = Vec::new();
        for (rank, solution) in results.pareto_solutions.iter().enumerate() {
            let profile_id = generate_profile_id(&solution.profile);
            profiles.push(ProfileEntry {
                profile: solution.profile.clone(),
                fitness: solution.fitness.clone(),
                objective_value: solution.objective_value,
                pareto_rank: rank,
                profile_id,
            });
        }

        // Best profile entry
        let best_profile_id = generate_profile_id(&results.best_solution.profile);
        let best_profile = ProfileEntry {
            profile: results.best_solution.profile.clone(),
            fitness: results.best_solution.fitness.clone(),
            objective_value: results.best_solution.objective_value,
            pareto_rank: 0, // Best overall
            profile_id: best_profile_id,
        };

        Ok(ProfileCollection {
            metadata,
            profiles,
            best_profile,
        })
    }

    /// Save collection as JSON
    fn save_as_json<P: AsRef<Path>>(&self, collection: &ProfileCollection, path: P) -> Result<()> {
        let json = serde_json::to_string_pretty(collection)
            .context("Failed to serialize profiles to JSON")?;

        let mut file = File::create(&path)
            .with_context(|| format!("Failed to create file: {:?}", path.as_ref()))?;

        file.write_all(json.as_bytes())
            .with_context(|| format!("Failed to write JSON to: {:?}", path.as_ref()))?;

        Ok(())
    }

    /// Save collection as compact JSON format
    fn save_as_compact_json<P: AsRef<Path>>(
        &self,
        collection: &ProfileCollection,
        path: P,
    ) -> Result<()> {
        // Save as compact JSON (not pretty-printed)
        let json_data = serde_json::to_vec(&collection)?;
        std::fs::write(path.as_ref(), &json_data)
            .with_context(|| format!("Failed to write compact JSON file: {:?}", path.as_ref()))?;

        Ok(())
    }

    /// Save best profile separately
    fn save_best_profile<P: AsRef<Path>>(&self, profile: &ProfileEntry, path: P) -> Result<()> {
        let json =
            serde_json::to_string_pretty(profile).context("Failed to serialize best profile")?;

        let mut file = File::create(&path)
            .with_context(|| format!("Failed to create best profile file: {:?}", path.as_ref()))?;

        file.write_all(json.as_bytes())
            .with_context(|| format!("Failed to write best profile: {:?}", path.as_ref()))?;

        Ok(())
    }

    /// Load from JSON
    fn load_from_json<P: AsRef<Path>>(path: P) -> Result<ProfileCollection> {
        let mut file = File::open(&path)
            .with_context(|| format!("Failed to open file: {:?}", path.as_ref()))?;

        let mut contents = String::new();
        file.read_to_string(&mut contents)
            .with_context(|| format!("Failed to read file: {:?}", path.as_ref()))?;

        let collection: ProfileCollection =
            serde_json::from_str(&contents).context("Failed to deserialize JSON")?;

        Ok(collection)
    }

    /// Load from safetensors (currently implemented as JSON for simplicity)
    fn load_from_safetensors<P: AsRef<Path>>(path: P) -> Result<ProfileCollection> {
        let data = std::fs::read(&path)
            .with_context(|| format!("Failed to read safetensors file: {:?}", path.as_ref()))?;
        let collection: ProfileCollection =
            serde_json::from_slice(&data).context("Failed to deserialize safetensors data")?;
        Ok(collection)
    }
}

/// Generate unique profile identifier
fn generate_profile_id(profile: &QuantProfile) -> String {
    format!(
        "{}b_{}q_{}s_{:.3}c_{:.2}m_{:.0}t",
        profile.bit_width,
        profile.qjl_dim,
        profile.rotation_seed,
        profile.clipping_percentile,
        profile.scale_multiplier,
        profile.qjl_threshold as f64 * 1e6 // Scale for readability
    )
}

/// Utility for profile analysis and comparison
pub struct ProfileAnalyzer;

impl ProfileAnalyzer {
    /// Analyze profile collection and generate insights
    pub fn analyze_collection(collection: &ProfileCollection) -> ProfileAnalysis {
        let profiles = &collection.profiles;

        // Memory gain statistics
        let memory_gains: Vec<f64> = profiles.iter().map(|p| p.fitness.memory_gain).collect();
        let avg_memory_gain = memory_gains.iter().sum::<f64>() / memory_gains.len() as f64;
        let max_memory_gain = memory_gains
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        // Quality degradation statistics
        let quality_degradations: Vec<f64> = profiles
            .iter()
            .map(|p| p.fitness.delta_brier_lm + p.fitness.cosine_penalty)
            .collect();
        let avg_quality_degradation =
            quality_degradations.iter().sum::<f64>() / quality_degradations.len() as f64;
        let min_quality_degradation = quality_degradations
            .iter()
            .fold(f64::INFINITY, |a, &b| a.min(b));

        // Bit width distribution
        let mut bit_width_counts = HashMap::new();
        for profile in profiles {
            *bit_width_counts
                .entry(profile.profile.bit_width)
                .or_insert(0) += 1;
        }

        // QJL dimension distribution
        let mut qjl_dim_counts = HashMap::new();
        for profile in profiles {
            *qjl_dim_counts.entry(profile.profile.qjl_dim).or_insert(0) += 1;
        }

        ProfileAnalysis {
            total_profiles: profiles.len(),
            avg_memory_gain,
            max_memory_gain,
            avg_quality_degradation,
            min_quality_degradation,
            bit_width_distribution: bit_width_counts,
            qjl_dim_distribution: qjl_dim_counts,
        }
    }

    /// Find profiles with specific characteristics
    pub fn find_profiles_by_criteria(
        collection: &ProfileCollection,
        min_memory_gain: Option<f64>,
        max_quality_degradation: Option<f64>,
        preferred_bit_width: Option<u8>,
    ) -> Vec<&ProfileEntry> {
        collection
            .profiles
            .iter()
            .filter(|profile| {
                if let Some(min_gain) = min_memory_gain {
                    if profile.fitness.memory_gain < min_gain {
                        return false;
                    }
                }

                if let Some(max_degradation) = max_quality_degradation {
                    let total_degradation =
                        profile.fitness.delta_brier_lm + profile.fitness.cosine_penalty;
                    if total_degradation > max_degradation {
                        return false;
                    }
                }

                if let Some(bits) = preferred_bit_width {
                    if profile.profile.bit_width != bits {
                        return false;
                    }
                }

                true
            })
            .collect()
    }
}

/// Analysis results for profile collection
#[derive(Debug, Clone)]
pub struct ProfileAnalysis {
    pub total_profiles: usize,
    pub avg_memory_gain: f64,
    pub max_memory_gain: f64,
    pub avg_quality_degradation: f64,
    pub min_quality_degradation: f64,
    pub bit_width_distribution: HashMap<u8, usize>,
    pub qjl_dim_distribution: HashMap<usize, usize>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::{SearchResults, SearchStatistics};
    use tempfile::TempDir;

    fn create_test_results() -> SearchResults {
        let solution = crate::pareto::ParetoSolution {
            profile: QuantProfile {
                bit_width: 4,
                qjl_dim: 32,
                rotation_seed: 42,
                qjl_threshold: 0.0001, scale_mode: "per_token".to_string(), clipping_percentile: 0.95, scale_multiplier: 1.0,
            },
            fitness: FitnessMetrics {
                memory_gain: 0.6,
                delta_brier_lm: 0.02,
                cosine_penalty: 0.05,
                latency_penalty: 0.1,
            },
            objective_value: 0.15,
        };

        SearchResults {
            pareto_solutions: vec![solution.clone()],
            best_solution: solution,
            statistics: SearchStatistics {
                total_iterations: 100,
                discrete_configs_explored: 4,
                avg_cmaes_iterations: 25.0,
                total_time_ms: 5000.0,
                evaluations_per_second: 20.0,
            },
        }
    }

    #[test]
    fn test_profile_export_import() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let exporter = ProfileExporter::new(temp_dir.path())?;

        let results = create_test_results();
        let dataset_info = DatasetInfo {
            num_samples: 100,
            avg_seq_length: 256.0,
            source: Some("test_dataset".to_string()),
        };

        // Export profiles
        let export_path =
            exporter.export_results(&results, dataset_info, "test_config".to_string())?;

        // Import profiles
        let imported = ProfileExporter::import_profiles(&export_path)?;

        assert_eq!(imported.profiles.len(), 1);
        assert_eq!(imported.profiles[0].profile.bit_width, 4);
        assert_eq!(imported.profiles[0].profile.qjl_dim, 32);

        Ok(())
    }

    #[test]
    fn test_profile_serialization_roundtrip() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let json_path = temp_dir.path().join("test_profile.json");

        let profile_entry = ProfileEntry {
            profile: QuantProfile {
                bit_width: 3,
                qjl_dim: 64,
                rotation_seed: 137,
                qjl_threshold: 1e-5 as f32,
                scale_mode: "per_token".to_string(),
                clipping_percentile: 0.99,
                scale_multiplier: 1.5,
            },
            fitness: FitnessMetrics {
                memory_gain: 0.7,
                delta_brier_lm: 0.01,
                cosine_penalty: 0.03,
                latency_penalty: 0.05,
            },
            objective_value: 0.12,
            pareto_rank: 0,
            profile_id: "test_profile".to_string(),
        };

        // Save to JSON
        let json_str = serde_json::to_string_pretty(&profile_entry)?;
        std::fs::write(&json_path, json_str)?;

        // Load from JSON
        let loaded_json = std::fs::read_to_string(&json_path)?;
        let loaded_profile: ProfileEntry = serde_json::from_str(&loaded_json)?;

        assert_eq!(loaded_profile.profile.bit_width, 3);
        assert_eq!(loaded_profile.profile.qjl_dim, 64);
        assert!((loaded_profile.fitness.memory_gain - 0.7).abs() < 1e-10);

        Ok(())
    }
}
