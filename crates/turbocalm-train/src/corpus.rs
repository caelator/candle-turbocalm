use std::collections::{BTreeMap, BTreeSet};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::pairs::{self, Corpus};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CorpusEntry {
    pub text: String,
    pub category: String,
    pub timestamp: i64,
    pub source: String,
}

pub fn load_from_jsonl<P: AsRef<Path>>(path: P) -> Result<Vec<CorpusEntry>> {
    let path = path.as_ref();
    let file = std::fs::File::open(path)
        .with_context(|| format!("failed to open corpus JSONL at {}", path.display()))?;
    let reader = BufReader::new(file);

    let mut entries = Vec::new();
    for (line_index, line) in reader.lines().enumerate() {
        let line = line.with_context(|| {
            format!(
                "failed to read line {} from {}",
                line_index + 1,
                path.display()
            )
        })?;
        if line.trim().is_empty() {
            continue;
        }
        let entry = serde_json::from_str::<CorpusEntry>(&line).with_context(|| {
            format!(
                "failed to parse JSONL line {} from {}",
                line_index + 1,
                path.display()
            )
        })?;
        entries.push(entry);
    }

    Ok(entries)
}

pub fn save_to_jsonl<P: AsRef<Path>>(entries: &[CorpusEntry], path: P) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }

    let file = std::fs::File::create(path)
        .with_context(|| format!("failed to create corpus JSONL at {}", path.display()))?;
    let mut writer = BufWriter::new(file);

    for entry in entries {
        let line = serde_json::to_string(entry).context("failed to serialize corpus entry")?;
        writer
            .write_all(line.as_bytes())
            .with_context(|| format!("failed to write {}", path.display()))?;
        writer
            .write_all(b"\n")
            .with_context(|| format!("failed to write newline to {}", path.display()))?;
    }

    writer
        .flush()
        .with_context(|| format!("failed to flush {}", path.display()))?;
    Ok(())
}

pub fn build_pairs_from_entries(entries: &[CorpusEntry]) -> Corpus {
    let categorized = pairs::from_categorized_texts(
        entries
            .iter()
            .map(|entry| (entry.text.clone(), entry.category.clone()))
            .collect(),
    );

    let mut temporal_corpora = Vec::new();
    let mut by_source = BTreeMap::<String, Vec<(String, i64)>>::new();
    for entry in entries {
        by_source
            .entry(entry.source.clone())
            .or_default()
            .push((entry.text.clone(), entry.timestamp));
    }
    for source_entries in by_source.values().cloned() {
        temporal_corpora.push(pairs::from_temporal_texts(source_entries));
    }

    let mut merged = pairs::merge_corpora(
        &std::iter::once(categorized.clone())
            .chain(temporal_corpora.iter().cloned())
            .collect::<Vec<_>>(),
    );
    merged.metadata.category_count = entries
        .iter()
        .map(|entry| entry.category.clone())
        .collect::<BTreeSet<_>>()
        .len();
    merged.metadata.source_count = by_source.len();
    merged.metadata.categorized_pair_count = categorized.metadata.categorized_pair_count;
    merged.metadata.temporal_pair_count = temporal_corpora
        .iter()
        .map(|corpus| corpus.metadata.temporal_pair_count)
        .sum();
    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_pairs_combines_categorized_and_temporal_signals() {
        let entries = vec![
            CorpusEntry {
                text: "alpha one".to_string(),
                category: "alpha".to_string(),
                timestamp: 1,
                source: "notes".to_string(),
            },
            CorpusEntry {
                text: "alpha two".to_string(),
                category: "alpha".to_string(),
                timestamp: 2,
                source: "notes".to_string(),
            },
            CorpusEntry {
                text: "beta one".to_string(),
                category: "beta".to_string(),
                timestamp: 10,
                source: "other".to_string(),
            },
        ];

        let corpus = build_pairs_from_entries(&entries);
        assert_eq!(corpus.metadata.category_count, 2);
        assert_eq!(corpus.metadata.source_count, 2);
        assert!(!corpus.pairs.is_empty());
    }
}
