use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::embedding::EmbeddingModel;
use crate::eval::EvalCorpus;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "kebab-case")]
pub enum SimilarityTier {
    Exact,
    NearDup,
    Related,
    Unrelated,
}

impl SimilarityTier {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::NearDup => "near-dup",
            Self::Related => "related",
            Self::Unrelated => "unrelated",
        }
    }

    fn ordered() -> [Self; 4] {
        [Self::Exact, Self::NearDup, Self::Related, Self::Unrelated]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CalibrationPair {
    pub left: String,
    pub right: String,
    pub tier: SimilarityTier,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CalibrationCorpus {
    pub pairs: Vec<CalibrationPair>,
}

impl CalibrationCorpus {
    pub fn from_json_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read calibration corpus {}", path.display()))?;
        match serde_json::from_str::<CalibrationCorpus>(&raw) {
            Ok(corpus) => Ok(corpus),
            Err(_) => {
                let eval = serde_json::from_str::<EvalCorpus>(&raw).with_context(|| {
                    format!("failed to parse calibration corpus {}", path.display())
                })?;
                Ok(Self::from_eval_corpus(&eval))
            }
        }
    }

    pub fn from_eval_corpus(eval: &EvalCorpus) -> Self {
        let mut pairs = Vec::new();
        let documents = eval.documents.iter().cloned().collect::<BTreeMap<_, _>>();

        for (_, document) in &eval.documents {
            pairs.push(CalibrationPair {
                left: document.clone(),
                right: document.clone(),
                tier: SimilarityTier::Exact,
            });
        }

        for pair in &eval.pairs {
            let mut relevant_texts = Vec::new();
            for relevant_id in &pair.relevant_ids {
                if let Some(text) = documents.get(relevant_id) {
                    relevant_texts.push(text.clone());
                    pairs.push(CalibrationPair {
                        left: pair.query.clone(),
                        right: text.clone(),
                        tier: SimilarityTier::NearDup,
                    });
                }
            }

            if relevant_texts.len() >= 2 {
                for left_index in 0..relevant_texts.len() {
                    for right_index in (left_index + 1)..relevant_texts.len() {
                        pairs.push(CalibrationPair {
                            left: relevant_texts[left_index].clone(),
                            right: relevant_texts[right_index].clone(),
                            tier: SimilarityTier::Related,
                        });
                    }
                }
            }

            for (doc_id, text) in &eval.documents {
                if pair.relevant_ids.contains(doc_id) {
                    continue;
                }
                pairs.push(CalibrationPair {
                    left: pair.query.clone(),
                    right: text.clone(),
                    tier: SimilarityTier::Unrelated,
                });
            }
        }

        Self { pairs }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CalibrationStats {
    pub count: usize,
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub stddev: f32,
    pub p10: f32,
    pub p25: f32,
    pub p50: f32,
    pub p75: f32,
    pub p90: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RecommendedThresholds {
    pub dedup: f32,
    pub cluster: f32,
    pub convergence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CalibrationReport {
    pub pair_count: usize,
    pub tiers: BTreeMap<SimilarityTier, CalibrationStats>,
    pub recommended: RecommendedThresholds,
}

pub fn run_calibration(
    model: &EmbeddingModel,
    corpus: &CalibrationCorpus,
) -> Result<CalibrationReport> {
    if corpus.pairs.is_empty() {
        bail!("calibration corpus must contain at least one pair")
    }

    let mut unique_texts = Vec::new();
    let mut text_to_index = BTreeMap::<String, usize>::new();
    for pair in &corpus.pairs {
        for text in [&pair.left, &pair.right] {
            if text_to_index.contains_key(text) {
                continue;
            }
            let index = unique_texts.len();
            unique_texts.push(text.clone());
            text_to_index.insert(text.clone(), index);
        }
    }

    let embeddings = model.embed_texts_pooled(&unique_texts)?;
    let mut scores = BTreeMap::<SimilarityTier, Vec<f32>>::new();
    for pair in &corpus.pairs {
        let left = text_to_index
            .get(&pair.left)
            .copied()
            .context("missing left embedding during calibration")?;
        let right = text_to_index
            .get(&pair.right)
            .copied()
            .context("missing right embedding during calibration")?;
        scores
            .entry(pair.tier)
            .or_default()
            .push(cosine_similarity(&embeddings[left], &embeddings[right]));
    }

    let mut tiers = BTreeMap::new();
    for tier in SimilarityTier::ordered() {
        let tier_scores = scores.remove(&tier).unwrap_or_default();
        tiers.insert(tier, compute_stats(&tier_scores));
    }

    let recommended = recommend_thresholds(&tiers);
    Ok(CalibrationReport {
        pair_count: corpus.pairs.len(),
        tiers,
        recommended,
    })
}

pub fn recommend_thresholds(
    tiers: &BTreeMap<SimilarityTier, CalibrationStats>,
) -> RecommendedThresholds {
    let exact = tiers.get(&SimilarityTier::Exact);
    let near_dup = tiers.get(&SimilarityTier::NearDup);
    let related = tiers.get(&SimilarityTier::Related);
    let unrelated = tiers.get(&SimilarityTier::Unrelated);

    let mut cluster = choose_threshold(
        related.map(|stats| stats.p10),
        unrelated.map(|stats| stats.p90),
        0.08,
    );
    let mut convergence = choose_threshold(
        related.map(|stats| stats.p25),
        unrelated.map(|stats| stats.p90),
        0.11,
    );
    let mut dedup = choose_threshold(
        near_dup
            .map(|stats| stats.p25)
            .or_else(|| exact.map(|stats| stats.p10)),
        related
            .map(|stats| stats.p75)
            .or_else(|| unrelated.map(|stats| stats.p90)),
        0.24,
    );

    cluster = cluster.clamp(-1.0, 1.0);
    convergence = convergence.clamp(-1.0, 1.0);
    dedup = dedup.clamp(-1.0, 1.0);

    if convergence <= cluster {
        convergence = (cluster + 0.01).clamp(-1.0, 1.0);
    }
    if dedup <= convergence {
        dedup = (convergence + 0.01).clamp(-1.0, 1.0);
    }
    if dedup - cluster < 0.02 {
        dedup = (cluster + 0.02).clamp(-1.0, 1.0);
        convergence = ((cluster + dedup) / 2.0).clamp(-1.0, 1.0);
    }

    RecommendedThresholds {
        dedup,
        cluster,
        convergence,
    }
}

pub fn save_calibration_toml<P: AsRef<Path>>(report: &CalibrationReport, path: P) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }

    let mut output = String::new();
    output.push_str("# Auto-generated by turbocalm-train calibrate\n");
    output.push_str(&format!("pair_count = {}\n\n", report.pair_count));
    output.push_str("[thresholds]\n");
    output.push_str(&format!("dedup = {:.6}\n", report.recommended.dedup));
    output.push_str(&format!("cluster = {:.6}\n", report.recommended.cluster));
    output.push_str(&format!(
        "convergence = {:.6}\n\n",
        report.recommended.convergence
    ));

    for tier in SimilarityTier::ordered() {
        let stats = report
            .tiers
            .get(&tier)
            .expect("every tier should have a stats entry");
        output.push_str(&format!("[tiers.{}]\n", tier.as_str()));
        output.push_str(&format!("count = {}\n", stats.count));
        output.push_str(&format!("min = {:.6}\n", stats.min));
        output.push_str(&format!("max = {:.6}\n", stats.max));
        output.push_str(&format!("mean = {:.6}\n", stats.mean));
        output.push_str(&format!("stddev = {:.6}\n", stats.stddev));
        output.push_str(&format!("p10 = {:.6}\n", stats.p10));
        output.push_str(&format!("p25 = {:.6}\n", stats.p25));
        output.push_str(&format!("p50 = {:.6}\n", stats.p50));
        output.push_str(&format!("p75 = {:.6}\n", stats.p75));
        output.push_str(&format!("p90 = {:.6}\n\n", stats.p90));
    }

    std::fs::write(path, output.as_bytes())
        .with_context(|| format!("failed to write {}", path.display()))
}

fn choose_threshold(high: Option<f32>, low: Option<f32>, fallback: f32) -> f32 {
    match (high, low) {
        (Some(high), Some(low)) => (high + low) / 2.0,
        (Some(high), None) => high,
        (None, Some(low)) => low,
        (None, None) => fallback,
    }
}

fn compute_stats(scores: &[f32]) -> CalibrationStats {
    if scores.is_empty() {
        return CalibrationStats {
            count: 0,
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            stddev: 0.0,
            p10: 0.0,
            p25: 0.0,
            p50: 0.0,
            p75: 0.0,
            p90: 0.0,
        };
    }

    let mut values = scores.to_vec();
    values.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let variance = values
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f32>()
        / values.len() as f32;

    CalibrationStats {
        count: values.len(),
        min: *values.first().unwrap_or(&0.0),
        max: *values.last().unwrap_or(&0.0),
        mean,
        stddev: variance.sqrt(),
        p10: percentile(&values, 0.10),
        p25: percentile(&values, 0.25),
        p50: percentile(&values, 0.50),
        p75: percentile(&values, 0.75),
        p90: percentile(&values, 0.90),
    }
}

fn percentile(values: &[f32], percentile: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let last = values.len().saturating_sub(1);
    let index = ((last as f32) * percentile).round() as usize;
    values[index.min(last)]
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

    let denom = (left_norm.sqrt() * right_norm.sqrt()).max(1e-6);
    dot / denom
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use turbocalm_models::CalmAutoencoderConfig;

    #[test]
    fn calibration_runs_and_suggests_ordered_thresholds() -> Result<()> {
        let model = EmbeddingModel::random(CalmAutoencoderConfig::default(), Device::Cpu)?;
        let report = run_calibration(
            &model,
            &CalibrationCorpus {
                pairs: vec![
                    CalibrationPair {
                        left: "steady breathing practice".to_string(),
                        right: "steady breathing practice".to_string(),
                        tier: SimilarityTier::Exact,
                    },
                    CalibrationPair {
                        left: "steady breathing practice".to_string(),
                        right: "steady breathing session".to_string(),
                        tier: SimilarityTier::NearDup,
                    },
                    CalibrationPair {
                        left: "calm inhale focus".to_string(),
                        right: "relaxation breathing practice".to_string(),
                        tier: SimilarityTier::Related,
                    },
                    CalibrationPair {
                        left: "distributed systems tracing".to_string(),
                        right: "garden compost seedlings".to_string(),
                        tier: SimilarityTier::Unrelated,
                    },
                ],
            },
        )?;

        assert_eq!(report.pair_count, 4);
        assert!(report.recommended.dedup >= report.recommended.convergence);
        assert!(report.recommended.convergence >= report.recommended.cluster);
        Ok(())
    }

    #[test]
    fn eval_corpus_can_seed_calibration_pairs() {
        let eval = EvalCorpus {
            pairs: vec![crate::EvalPair {
                query: "steady breathing practice".to_string(),
                relevant_ids: vec!["doc-1".to_string()],
            }],
            documents: vec![
                ("doc-1".to_string(), "steady breathing practice".to_string()),
                ("doc-2".to_string(), "systems design".to_string()),
            ],
        };

        let corpus = CalibrationCorpus::from_eval_corpus(&eval);
        assert!(corpus
            .pairs
            .iter()
            .any(|pair| pair.tier == SimilarityTier::Exact));
        assert!(corpus
            .pairs
            .iter()
            .any(|pair| pair.tier == SimilarityTier::Unrelated));
    }
}
