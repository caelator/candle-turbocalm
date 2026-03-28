//! Calibration report generation
//!
//! Generates comprehensive human-readable reports of calibration results
//! with metrics, visualizations, and recommendations.

use crate::profiles::{ProfileAnalysis, ProfileAnalyzer, ProfileCollection, ProfileEntry};
use crate::search::SearchResults;
use crate::{FitnessMetrics, QuantProfile};
use anyhow::Result;
use std::fmt::Write as FmtWrite;
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Report generator for calibration results
pub struct ReportGenerator {
    /// Include detailed profile listings
    include_details: bool,
    /// Include recommendations
    include_recommendations: bool,
}

/// Generated calibration report
#[derive(Debug, Clone)]
pub struct CalibrationReport {
    /// Report content in markdown format
    pub content: String,
    /// Report metadata
    pub metadata: ReportMetadata,
}

/// Report metadata
#[derive(Debug, Clone)]
pub struct ReportMetadata {
    /// Report generation timestamp
    pub generated_at: String,
    /// Report version
    pub version: String,
    /// Number of profiles analyzed
    pub profiles_analyzed: usize,
}

impl ReportGenerator {
    /// Create new report generator
    pub fn new(include_details: bool, include_recommendations: bool) -> Self {
        Self {
            include_details,
            include_recommendations,
        }
    }

    /// Generate complete calibration report
    pub fn generate_report(&self, collection: &ProfileCollection) -> Result<CalibrationReport> {
        let mut content = String::new();

        // Generate report sections
        self.write_header(&mut content, collection)?;
        self.write_executive_summary(&mut content, collection)?;
        self.write_search_summary(&mut content, collection)?;
        self.write_pareto_analysis(&mut content, collection)?;
        self.write_best_profiles(&mut content, collection)?;

        if self.include_details {
            self.write_detailed_profiles(&mut content, collection)?;
        }

        if self.include_recommendations {
            self.write_recommendations(&mut content, collection)?;
        }

        self.write_appendix(&mut content, collection)?;

        let metadata = ReportMetadata {
            generated_at: chrono::Utc::now()
                .format("%Y-%m-%d %H:%M:%S UTC")
                .to_string(),
            version: "1.0".to_string(),
            profiles_analyzed: collection.profiles.len(),
        };

        Ok(CalibrationReport { content, metadata })
    }

    /// Save report to file
    pub fn save_report<P: AsRef<Path>>(&self, report: &CalibrationReport, path: P) -> Result<()> {
        let mut file = File::create(&path)?;
        file.write_all(report.content.as_bytes())?;
        tracing::info!("Saved calibration report to {:?}", path.as_ref());
        Ok(())
    }

    /// Write report header
    fn write_header(&self, content: &mut String, collection: &ProfileCollection) -> Result<()> {
        writeln!(content, "# TurboCalm Quantization Calibration Report")?;
        writeln!(content)?;
        writeln!(
            content,
            "**Generated:** {}",
            chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
        )?;
        writeln!(
            content,
            "**Calibration Run:** {}",
            collection.metadata.timestamp
        )?;
        writeln!(content, "**Profiles Found:** {}", collection.profiles.len())?;
        writeln!(content)?;
        writeln!(content, "---")?;
        writeln!(content)?;
        Ok(())
    }

    /// Write executive summary
    fn write_executive_summary(
        &self,
        content: &mut String,
        collection: &ProfileCollection,
    ) -> Result<()> {
        writeln!(content, "## Executive Summary")?;
        writeln!(content)?;

        let analysis = ProfileAnalyzer::analyze_collection(collection);
        let best = &collection.best_profile;

        writeln!(content, "### Key Findings")?;
        writeln!(content)?;
        writeln!(
            content,
            "- **Total Profiles Evaluated:** {}",
            analysis.total_profiles
        )?;
        writeln!(
            content,
            "- **Best Memory Reduction:** {:.1}%",
            best.fitness.memory_gain * 100.0
        )?;
        writeln!(
            content,
            "- **Quality Preservation:** {:.3} Brier increase, {:.3} cosine penalty",
            best.fitness.delta_brier_lm, best.fitness.cosine_penalty
        )?;
        writeln!(
            content,
            "- **Optimal Configuration:** {} bits, {} QJL dimensions",
            best.profile.bit_width, best.profile.qjl_dim
        )?;
        writeln!(content)?;

        // Performance summary
        writeln!(content, "### Performance Summary")?;
        writeln!(content)?;
        writeln!(content, "| Metric | Value |")?;
        writeln!(content, "|--------|-------|")?;
        writeln!(
            content,
            "| Search Time | {:.1}s |",
            collection.metadata.search_stats.search_time_seconds
        )?;
        writeln!(
            content,
            "| Evaluations | {} |",
            collection.metadata.search_stats.total_evaluations
        )?;
        writeln!(
            content,
            "| Avg Memory Gain | {:.1}% |",
            analysis.avg_memory_gain * 100.0
        )?;
        writeln!(
            content,
            "| Max Memory Gain | {:.1}% |",
            analysis.max_memory_gain * 100.0
        )?;
        writeln!(
            content,
            "| Avg Quality Loss | {:.3} |",
            analysis.avg_quality_degradation
        )?;
        writeln!(
            content,
            "| Min Quality Loss | {:.3} |",
            analysis.min_quality_degradation
        )?;
        writeln!(content)?;

        Ok(())
    }

    /// Write search summary
    fn write_search_summary(
        &self,
        content: &mut String,
        collection: &ProfileCollection,
    ) -> Result<()> {
        writeln!(content, "## Search Configuration & Results")?;
        writeln!(content)?;

        writeln!(content, "### Dataset Information")?;
        writeln!(content)?;
        writeln!(
            content,
            "- **Samples:** {}",
            collection.metadata.dataset_info.num_samples
        )?;
        writeln!(
            content,
            "- **Avg Sequence Length:** {:.1}",
            collection.metadata.dataset_info.avg_seq_length
        )?;
        if let Some(ref source) = collection.metadata.dataset_info.source {
            writeln!(content, "- **Source:** {}", source)?;
        }
        writeln!(content)?;

        writeln!(content, "### Search Strategy")?;
        writeln!(content)?;
        writeln!(content, "```")?;
        writeln!(content, "{}", collection.metadata.config_summary)?;
        writeln!(content, "```")?;
        writeln!(content)?;

        Ok(())
    }

    /// Write Pareto analysis
    fn write_pareto_analysis(
        &self,
        content: &mut String,
        collection: &ProfileCollection,
    ) -> Result<()> {
        writeln!(content, "## Pareto Front Analysis")?;
        writeln!(content)?;

        writeln!(content, "The evolutionary search discovered {} non-dominated solutions forming the Pareto frontier.",
                 collection.profiles.len())?;
        writeln!(content)?;

        // Trade-off analysis
        writeln!(content, "### Trade-off Analysis")?;
        writeln!(content)?;

        // Memory vs Quality scatter
        writeln!(content, "#### Memory Gain vs Quality Loss")?;
        writeln!(content)?;
        writeln!(
            content,
            "| Rank | Memory Gain (%) | Quality Loss | Bit Width | QJL Dim | Objective |"
        )?;
        writeln!(
            content,
            "|------|----------------|---------------|-----------|---------|-----------|"
        )?;

        for (i, profile) in collection.profiles.iter().take(10).enumerate() {
            let quality_loss = profile.fitness.delta_brier_lm + profile.fitness.cosine_penalty;
            writeln!(
                content,
                "| {} | {:.1}% | {:.4} | {} | {} | {:.4} |",
                i + 1,
                profile.fitness.memory_gain * 100.0,
                quality_loss,
                profile.profile.bit_width,
                profile.profile.qjl_dim,
                profile.objective_value
            )?;
        }

        if collection.profiles.len() > 10 {
            writeln!(content, "| ... | ... | ... | ... | ... | ... |")?;
            writeln!(
                content,
                "*Showing top 10 of {} solutions*",
                collection.profiles.len()
            )?;
        }
        writeln!(content)?;

        Ok(())
    }

    /// Write best profiles section
    fn write_best_profiles(
        &self,
        content: &mut String,
        collection: &ProfileCollection,
    ) -> Result<()> {
        writeln!(content, "## Recommended Profiles")?;
        writeln!(content)?;

        // Best overall
        writeln!(content, "### 🏆 Best Overall Profile")?;
        writeln!(content)?;
        self.write_profile_details(content, &collection.best_profile, "overall")?;
        writeln!(content)?;

        // Best memory gain
        let best_memory = collection.profiles.iter().max_by(|a, b| {
            a.fitness
                .memory_gain
                .partial_cmp(&b.fitness.memory_gain)
                .unwrap()
        });

        if let Some(profile) = best_memory {
            writeln!(content, "### 💾 Best Memory Reduction")?;
            writeln!(content)?;
            self.write_profile_details(content, profile, "memory")?;
            writeln!(content)?;
        }

        // Best quality preservation
        let best_quality = collection.profiles.iter().min_by(|a, b| {
            let quality_a = a.fitness.delta_brier_lm + a.fitness.cosine_penalty;
            let quality_b = b.fitness.delta_brier_lm + b.fitness.cosine_penalty;
            quality_a.partial_cmp(&quality_b).unwrap()
        });

        if let Some(profile) = best_quality {
            writeln!(content, "### 🎯 Best Quality Preservation")?;
            writeln!(content)?;
            self.write_profile_details(content, profile, "quality")?;
            writeln!(content)?;
        }

        Ok(())
    }

    /// Write profile details
    fn write_profile_details(
        &self,
        content: &mut String,
        profile: &ProfileEntry,
        category: &str,
    ) -> Result<()> {
        writeln!(content, "**Profile ID:** `{}`", profile.profile_id)?;
        writeln!(content)?;

        writeln!(content, "**Configuration:**")?;
        writeln!(content, "- Bit Width: {} bits", profile.profile.bit_width)?;
        writeln!(content, "- QJL Dimension: {}", profile.profile.qjl_dim)?;
        writeln!(
            content,
            "- Rotation Seed: {}",
            profile.profile.rotation_seed
        )?;
        writeln!(
            content,
            "- Clipping Percentile: {:.3}",
            profile.profile.clipping_percentile
        )?;
        writeln!(
            content,
            "- Scale Multiplier: {:.2}",
            profile.profile.scale_multiplier
        )?;
        writeln!(
            content,
            "- QJL Threshold: {:.1e}",
            profile.profile.qjl_threshold
        )?;
        writeln!(content)?;

        writeln!(content, "**Performance:**")?;
        writeln!(
            content,
            "- Memory Reduction: {:.1}%",
            profile.fitness.memory_gain * 100.0
        )?;
        writeln!(
            content,
            "- Brier Score Increase: {:.4}",
            profile.fitness.delta_brier_lm
        )?;
        writeln!(
            content,
            "- Cosine Similarity Loss: {:.4}",
            profile.fitness.cosine_penalty
        )?;
        writeln!(
            content,
            "- Latency Impact: {:.1}%",
            profile.fitness.latency_penalty * 100.0
        )?;
        writeln!(
            content,
            "- Weighted Objective: {:.4}",
            profile.objective_value
        )?;
        writeln!(content)?;

        // Usage code snippet
        writeln!(content, "**Usage:**")?;
        writeln!(content, "```rust")?;
        writeln!(content, "let profile = QuantProfile {{")?;
        writeln!(content, "    bit_width: {},", profile.profile.bit_width)?;
        writeln!(content, "    qjl_dim: {},", profile.profile.qjl_dim)?;
        writeln!(
            content,
            "    rotation_seed: {},",
            profile.profile.rotation_seed
        )?;
        writeln!(
            content,
            "    qjl_threshold: {},",
            profile.profile.qjl_threshold
        )?;
        writeln!(
            content,
            "    scale_mode: \"{}\",",
            profile.profile.scale_mode
        )?;
        writeln!(
            content,
            "    clipping_percentile: {},",
            profile.profile.clipping_percentile
        )?;
        writeln!(
            content,
            "    scale_multiplier: {},",
            profile.profile.scale_multiplier
        )?;
        writeln!(content, "}};")?;
        writeln!(content, "```")?;

        Ok(())
    }

    /// Write detailed profiles section
    fn write_detailed_profiles(
        &self,
        content: &mut String,
        collection: &ProfileCollection,
    ) -> Result<()> {
        writeln!(content, "## Detailed Profile Listings")?;
        writeln!(content)?;

        writeln!(content, "### All Pareto-Optimal Solutions")?;
        writeln!(content)?;

        for (i, profile) in collection.profiles.iter().enumerate() {
            writeln!(content, "#### Profile {} - {}", i + 1, profile.profile_id)?;
            writeln!(content)?;

            writeln!(content, "| Parameter | Value |")?;
            writeln!(content, "|-----------|-------|")?;
            writeln!(content, "| Bit Width | {} |", profile.profile.bit_width)?;
            writeln!(content, "| QJL Dimension | {} |", profile.profile.qjl_dim)?;
            writeln!(
                content,
                "| Rotation Seed | {} |",
                profile.profile.rotation_seed
            )?;
            writeln!(
                content,
                "| Clipping Percentile | {:.3} |",
                profile.profile.clipping_percentile
            )?;
            writeln!(
                content,
                "| Scale Multiplier | {:.2} |",
                profile.profile.scale_multiplier
            )?;
            writeln!(
                content,
                "| QJL Threshold | {:.1e} |",
                profile.profile.qjl_threshold
            )?;
            writeln!(
                content,
                "| **Memory Gain** | **{:.1}%** |",
                profile.fitness.memory_gain * 100.0
            )?;
            writeln!(
                content,
                "| **Quality Loss** | **{:.4}** |",
                profile.fitness.delta_brier_lm + profile.fitness.cosine_penalty
            )?;
            writeln!(
                content,
                "| **Objective Value** | **{:.4}** |",
                profile.objective_value
            )?;
            writeln!(content)?;
        }

        Ok(())
    }

    /// Write recommendations
    fn write_recommendations(
        &self,
        content: &mut String,
        collection: &ProfileCollection,
    ) -> Result<()> {
        writeln!(content, "## Recommendations")?;
        writeln!(content)?;

        let analysis = ProfileAnalyzer::analyze_collection(collection);

        writeln!(content, "### Deployment Guidelines")?;
        writeln!(content)?;

        // Production recommendations based on analysis
        if analysis.max_memory_gain > 0.5 {
            writeln!(content, "✅ **Production Ready:** Significant memory savings ({:.1}%) with acceptable quality trade-offs.",
                     analysis.max_memory_gain * 100.0)?;
        } else {
            writeln!(content, "⚠️  **Consider Alternatives:** Limited memory savings ({:.1}%) may not justify deployment overhead.",
                     analysis.max_memory_gain * 100.0)?;
        }
        writeln!(content)?;

        writeln!(content, "### Configuration Recommendations")?;
        writeln!(content)?;

        // Bit width recommendations
        let dominant_bits = analysis
            .bit_width_distribution
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(&bits, _)| bits);

        if let Some(bits) = dominant_bits {
            writeln!(
                content,
                "- **Preferred Bit Width:** {} bits (appears in {}% of optimal solutions)",
                bits,
                analysis.bit_width_distribution[&bits] * 100 / analysis.total_profiles
            )?;
        }

        // QJL dimension recommendations
        let dominant_qjl = analysis
            .qjl_dim_distribution
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(&dim, _)| dim);

        if let Some(dim) = dominant_qjl {
            writeln!(
                content,
                "- **Preferred QJL Dimension:** {} (appears in {}% of optimal solutions)",
                dim,
                analysis.qjl_dim_distribution[&dim] * 100 / analysis.total_profiles
            )?;
        }
        writeln!(content)?;

        writeln!(content, "### Next Steps")?;
        writeln!(content)?;
        writeln!(
            content,
            "1. **Validation:** Test recommended profiles on held-out validation sets"
        )?;
        writeln!(
            content,
            "2. **A/B Testing:** Compare performance in production environment"
        )?;
        writeln!(
            content,
            "3. **Monitoring:** Track memory usage and quality metrics post-deployment"
        )?;
        writeln!(
            content,
            "4. **Iteration:** Re-run calibration with updated datasets as needed"
        )?;
        writeln!(content)?;

        Ok(())
    }

    /// Write appendix
    fn write_appendix(&self, content: &mut String, collection: &ProfileCollection) -> Result<()> {
        writeln!(content, "## Appendix")?;
        writeln!(content)?;

        writeln!(content, "### Methodology")?;
        writeln!(content)?;
        writeln!(
            content,
            "This calibration used a two-level evolutionary optimization strategy:"
        )?;
        writeln!(content)?;
        writeln!(content, "1. **Outer Loop:** Discrete parameter enumeration (bit width, QJL dimension, rotation seed)")?;
        writeln!(content, "2. **Inner Loop:** CMA-ES optimization of continuous parameters (clipping, scaling, thresholds)")?;
        writeln!(content, "3. **Multi-objective:** Balanced memory reduction, quality preservation, and latency considerations")?;
        writeln!(content)?;

        writeln!(content, "### Objective Function")?;
        writeln!(content)?;
        writeln!(content, "```")?;
        writeln!(
            content,
            "objective = memory_gain - λ₁·ΔBrierLM - λ₂·cosine_penalty - λ₃·latency_penalty"
        )?;
        writeln!(content, "```")?;
        writeln!(content)?;

        writeln!(content, "### Metrics Definitions")?;
        writeln!(content)?;
        writeln!(
            content,
            "- **Memory Gain:** Fractional reduction in model memory footprint"
        )?;
        writeln!(
            content,
            "- **ΔBrierLM:** Change in Brier score for language modeling task"
        )?;
        writeln!(content, "- **Cosine Penalty:** 1 - cosine_similarity between original and quantized activations")?;
        writeln!(
            content,
            "- **Latency Penalty:** Fractional increase in inference latency"
        )?;
        writeln!(content)?;

        writeln!(content, "### Report Generation")?;
        writeln!(content)?;
        writeln!(content, "Generated by TurboCalm Calibration Engine v1.0")?;
        writeln!(content, "Report Template: Standard Calibration Report")?;
        writeln!(
            content,
            "Generated: {}",
            chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
        )?;
        writeln!(content)?;

        Ok(())
    }
}

/// Utility functions for report customization
pub struct ReportCustomizer;

impl ReportCustomizer {
    /// Generate executive-only summary report
    pub fn generate_executive_summary(collection: &ProfileCollection) -> Result<String> {
        let generator = ReportGenerator::new(false, false);
        let report = generator.generate_report(collection)?;

        // Extract only executive summary section
        let lines: Vec<&str> = report.content.lines().collect();
        let start_idx = lines
            .iter()
            .position(|&line| line == "## Executive Summary");
        let end_idx = lines
            .iter()
            .position(|&line| line.starts_with("## ") && line != "## Executive Summary");

        if let (Some(start), Some(end)) = (start_idx, end_idx) {
            Ok(lines[start..end].join("\n"))
        } else {
            Ok(report.content)
        }
    }

    /// Generate technical-only report
    pub fn generate_technical_report(collection: &ProfileCollection) -> Result<String> {
        let generator = ReportGenerator::new(true, false);
        let report = generator.generate_report(collection)?;
        Ok(report.content)
    }

    /// Generate comparison report for multiple calibration runs
    pub fn generate_comparison_report(
        collections: &[(&str, &ProfileCollection)],
    ) -> Result<String> {
        let mut content = String::new();

        writeln!(content, "# TurboCalm Calibration Comparison Report")?;
        writeln!(content)?;
        writeln!(
            content,
            "**Generated:** {}",
            chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
        )?;
        writeln!(content, "**Runs Compared:** {}", collections.len())?;
        writeln!(content)?;

        writeln!(content, "## Summary Comparison")?;
        writeln!(content)?;

        writeln!(
            content,
            "| Run | Profiles | Best Memory Gain | Best Quality Loss | Search Time |"
        )?;
        writeln!(
            content,
            "|-----|----------|------------------|-------------------|-------------|"
        )?;

        for (name, collection) in collections {
            let best = &collection.best_profile;
            let quality_loss = best.fitness.delta_brier_lm + best.fitness.cosine_penalty;

            writeln!(
                content,
                "| {} | {} | {:.1}% | {:.4} | {:.1}s |",
                name,
                collection.profiles.len(),
                best.fitness.memory_gain * 100.0,
                quality_loss,
                collection.metadata.search_stats.search_time_seconds
            )?;
        }

        writeln!(content)?;

        Ok(content)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profiles::{CalibrationMetadata, DatasetInfo, SearchStats};
    use crate::ContinuousParams;

    fn create_test_collection() -> ProfileCollection {
        let profile_entry = ProfileEntry {
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
            pareto_rank: 0,
            profile_id: "test_profile".to_string(),
        };

        ProfileCollection {
            metadata: CalibrationMetadata {
                timestamp: "20240101_120000".to_string(),
                config_summary: "Test configuration".to_string(),
                dataset_info: DatasetInfo {
                    num_samples: 100,
                    avg_seq_length: 256.0,
                    source: Some("test".to_string()),
                },
                search_stats: SearchStats {
                    total_evaluations: 500,
                    pareto_solutions: 1,
                    search_time_seconds: 60.0,
                },
            },
            profiles: vec![profile_entry.clone()],
            best_profile: profile_entry,
        }
    }

    #[test]
    fn test_report_generation() -> Result<()> {
        let collection = create_test_collection();
        let generator = ReportGenerator::new(true, true);
        let report = generator.generate_report(&collection)?;

        assert!(report
            .content
            .contains("# TurboCalm Quantization Calibration Report"));
        assert!(report.content.contains("## Executive Summary"));
        assert!(report.content.contains("## Recommendations"));
        assert_eq!(report.metadata.profiles_analyzed, 1);

        Ok(())
    }

    #[test]
    fn test_executive_summary_only() -> Result<()> {
        let collection = create_test_collection();
        let summary = ReportCustomizer::generate_executive_summary(&collection)?;

        assert!(summary.contains("## Executive Summary"));
        assert!(!summary.contains("## Detailed Profile Listings"));

        Ok(())
    }
}
