use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::embedding::EmbeddingModel;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EvalPair {
    pub query: String,
    pub relevant_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EvalCorpus {
    pub pairs: Vec<EvalPair>,
    pub documents: Vec<(String, String)>,
}

impl EvalCorpus {
    pub fn from_json_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read eval corpus {}", path.display()))?;
        serde_json::from_str(&raw)
            .with_context(|| format!("failed to parse eval corpus {}", path.display()))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EvalMetrics {
    pub recall_at_5: f32,
    pub mrr: f32,
    pub avg_cosine_similar: f32,
    pub avg_cosine_dissimilar: f32,
}

impl EvalMetrics {
    pub fn render_table(&self) -> String {
        format!(
            "+------------------------+----------+\n\
             | Metric                 | Value    |\n\
             +------------------------+----------+\n\
             | recall@5               | {recall:>8.4} |\n\
             | mrr                    | {mrr:>8.4} |\n\
             | avg_cosine_similar     | {similar:>8.4} |\n\
             | avg_cosine_dissimilar  | {dissimilar:>8.4} |\n\
             +------------------------+----------+",
            recall = self.recall_at_5,
            mrr = self.mrr,
            similar = self.avg_cosine_similar,
            dissimilar = self.avg_cosine_dissimilar,
        )
    }
}

pub fn run_eval(model: &EmbeddingModel, corpus: &EvalCorpus) -> Result<EvalMetrics> {
    if corpus.pairs.is_empty() {
        bail!("eval corpus must contain at least one query pair")
    }
    if corpus.documents.is_empty() {
        bail!("eval corpus must contain at least one document")
    }

    let mut seen_ids = BTreeSet::new();
    let mut doc_ids = Vec::with_capacity(corpus.documents.len());
    let mut doc_texts = Vec::with_capacity(corpus.documents.len());
    for (id, text) in &corpus.documents {
        if !seen_ids.insert(id.clone()) {
            bail!("duplicate document id in eval corpus: {id}")
        }
        doc_ids.push(id.clone());
        doc_texts.push(text.clone());
    }

    let doc_embeddings = model.embed_texts_pooled(&doc_texts)?;
    let doc_by_id = doc_ids
        .iter()
        .cloned()
        .zip(doc_embeddings.iter().cloned())
        .collect::<BTreeMap<_, _>>();

    let query_texts = corpus
        .pairs
        .iter()
        .map(|pair| pair.query.clone())
        .collect::<Vec<_>>();
    let query_embeddings = model.embed_texts_pooled(&query_texts)?;

    let mut recall_sum = 0.0f32;
    let mut mrr_sum = 0.0f32;
    let mut similar_sum = 0.0f32;
    let mut similar_count = 0usize;
    let mut dissimilar_sum = 0.0f32;
    let mut dissimilar_count = 0usize;

    for (pair, query_embedding) in corpus.pairs.iter().zip(query_embeddings.iter()) {
        if pair.relevant_ids.is_empty() {
            bail!("eval pair {:?} has no relevant_ids", pair.query)
        }

        let relevant_ids = pair.relevant_ids.iter().cloned().collect::<BTreeSet<_>>();
        let mut ranked = doc_ids
            .iter()
            .map(|doc_id| {
                let doc_embedding = doc_by_id
                    .get(doc_id)
                    .expect("document embeddings should exist for every id");
                let score = cosine_similarity(query_embedding, doc_embedding);
                (doc_id.clone(), score)
            })
            .collect::<Vec<_>>();

        ranked.sort_by(|left, right| {
            right
                .1
                .partial_cmp(&left.1)
                .unwrap_or(Ordering::Equal)
        });

        let top_k = ranked.iter().take(5);
        let hits = top_k
            .clone()
            .filter(|(doc_id, _)| relevant_ids.contains(doc_id))
            .count();
        recall_sum += hits as f32 / relevant_ids.len() as f32;

        if let Some((rank, _)) = ranked
            .iter()
            .enumerate()
            .find(|(_, (doc_id, _))| relevant_ids.contains(doc_id))
        {
            mrr_sum += 1.0 / (rank as f32 + 1.0);
        }

        for (doc_id, score) in ranked {
            if relevant_ids.contains(&doc_id) {
                similar_sum += score;
                similar_count += 1;
            } else {
                dissimilar_sum += score;
                dissimilar_count += 1;
            }
        }
    }

    Ok(EvalMetrics {
        recall_at_5: recall_sum / corpus.pairs.len() as f32,
        mrr: mrr_sum / corpus.pairs.len() as f32,
        avg_cosine_similar: average(similar_sum, similar_count),
        avg_cosine_dissimilar: average(dissimilar_sum, dissimilar_count),
    })
}

fn average(sum: f32, count: usize) -> f32 {
    if count == 0 {
        0.0
    } else {
        sum / count as f32
    }
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
    fn eval_returns_finite_metrics() -> Result<()> {
        let model = EmbeddingModel::random(CalmAutoencoderConfig::default(), Device::Cpu)?;
        let corpus = EvalCorpus {
            pairs: vec![EvalPair {
                query: "steady breathing practice".to_string(),
                relevant_ids: vec!["doc-1".to_string()],
            }],
            documents: vec![
                ("doc-1".to_string(), "steady breathing practice".to_string()),
                ("doc-2".to_string(), "systems architecture".to_string()),
            ],
        };

        let metrics = run_eval(&model, &corpus)?;
        assert!(metrics.recall_at_5.is_finite());
        assert!(metrics.mrr.is_finite());
        Ok(())
    }
}
