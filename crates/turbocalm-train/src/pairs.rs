use std::collections::{BTreeMap, BTreeSet, HashSet};

use rand::seq::SliceRandom;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TrainingPair {
    pub anchor: String,
    pub positive: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CorpusMetadata {
    pub pair_count: usize,
    pub category_count: usize,
    pub source_count: usize,
    pub categorized_pair_count: usize,
    pub temporal_pair_count: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Corpus {
    pub pairs: Vec<TrainingPair>,
    pub metadata: CorpusMetadata,
}

impl Corpus {
    pub fn new(pairs: Vec<TrainingPair>, metadata: CorpusMetadata) -> Self {
        Self { pairs, metadata }
    }
}

pub fn from_categorized_texts(texts: Vec<(String, String)>) -> Corpus {
    let mut grouped = BTreeMap::<String, Vec<String>>::new();
    for (text, category) in texts {
        grouped.entry(category).or_default().push(text);
    }

    let mut pairs = Vec::new();
    for entries in grouped.values() {
        for left in 0..entries.len() {
            for right in (left + 1)..entries.len() {
                if let Some(pair) = build_pair(entries[left].clone(), entries[right].clone()) {
                    pairs.push(pair);
                }
            }
        }
    }

    let pair_count = pairs.len();
    Corpus::new(
        pairs,
        CorpusMetadata {
            pair_count,
            category_count: grouped.len(),
            source_count: 0,
            categorized_pair_count: pair_count,
            temporal_pair_count: 0,
        },
    )
}

pub fn from_temporal_texts(texts: Vec<(String, i64)>) -> Corpus {
    let mut ordered = texts;
    ordered.sort_by_key(|(_, timestamp)| *timestamp);

    let mut pairs = Vec::new();
    for window in ordered.windows(2) {
        let (anchor, _) = &window[0];
        let (positive, _) = &window[1];
        if let Some(pair) = build_pair(anchor.clone(), positive.clone()) {
            pairs.push(pair);
        }
    }

    let pair_count = pairs.len();
    Corpus::new(
        pairs,
        CorpusMetadata {
            pair_count,
            category_count: 0,
            source_count: 0,
            categorized_pair_count: 0,
            temporal_pair_count: pair_count,
        },
    )
}

pub fn merge_corpora(corpora: &[Corpus]) -> Corpus {
    let mut seen = HashSet::<(String, String)>::new();
    let mut merged_pairs = Vec::new();
    let mut categories = BTreeSet::new();
    let mut source_count = 0usize;
    let mut categorized_pair_count = 0usize;
    let mut temporal_pair_count = 0usize;

    for corpus in corpora {
        categorized_pair_count += corpus.metadata.categorized_pair_count;
        temporal_pair_count += corpus.metadata.temporal_pair_count;
        source_count += corpus.metadata.source_count;
        if corpus.metadata.category_count > 0 {
            categories.insert(corpus.metadata.category_count);
        }

        for pair in &corpus.pairs {
            let key = canonical_pair_key(pair);
            if seen.insert(key) {
                merged_pairs.push(pair.clone());
            }
        }
    }

    let pair_count = merged_pairs.len();
    Corpus::new(
        merged_pairs,
        CorpusMetadata {
            pair_count,
            category_count: categories.into_iter().max().unwrap_or_default(),
            source_count,
            categorized_pair_count,
            temporal_pair_count,
        },
    )
}

pub fn generate_epoch_batches(corpus: &Corpus, batch_size: usize) -> Vec<Vec<TrainingPair>> {
    if batch_size == 0 {
        return Vec::new();
    }

    let mut shuffled = corpus.pairs.clone();
    shuffled.shuffle(&mut rand::thread_rng());

    shuffled
        .chunks(batch_size)
        .map(|chunk| chunk.to_vec())
        .collect()
}

fn build_pair(anchor: String, positive: String) -> Option<TrainingPair> {
    if anchor.trim().is_empty() || positive.trim().is_empty() || anchor == positive {
        return None;
    }
    Some(TrainingPair { anchor, positive })
}

fn canonical_pair_key(pair: &TrainingPair) -> (String, String) {
    if pair.anchor <= pair.positive {
        (pair.anchor.clone(), pair.positive.clone())
    } else {
        (pair.positive.clone(), pair.anchor.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn categorized_pairs_are_generated_within_groups() {
        let corpus = from_categorized_texts(vec![
            ("a".to_string(), "x".to_string()),
            ("b".to_string(), "x".to_string()),
            ("c".to_string(), "x".to_string()),
            ("d".to_string(), "y".to_string()),
        ]);

        assert_eq!(corpus.pairs.len(), 3);
    }

    #[test]
    fn temporal_pairs_follow_sorted_adjacency() {
        let corpus = from_temporal_texts(vec![
            ("later".to_string(), 3),
            ("first".to_string(), 1),
            ("middle".to_string(), 2),
        ]);

        assert_eq!(
            corpus.pairs,
            vec![
                TrainingPair {
                    anchor: "first".to_string(),
                    positive: "middle".to_string()
                },
                TrainingPair {
                    anchor: "middle".to_string(),
                    positive: "later".to_string()
                }
            ]
        );
    }
}
