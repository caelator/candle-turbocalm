use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsStr;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{anyhow, bail, Context, Result};
use chrono::DateTime;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::pairs::{self, Corpus};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CorpusEntry {
    pub text: String,
    pub category: String,
    pub timestamp: i64,
    pub source: String,
}

#[derive(Debug, Deserialize)]
struct MemoryExportRow {
    text: String,
    category: String,
    scope: Option<String>,
    timestamp: Option<i64>,
    metadata: Option<String>,
}

const LANCEDB_READER_SCRIPT: &str = r#"
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";

const pluginRoot = process.env.LANCEDB_PLUGIN_ROOT;
const dbPath = process.env.LANCEDB_DB_PATH;
const tableName = process.env.LANCEDB_TABLE_NAME || "memories";

if (!pluginRoot || !dbPath) {
  throw new Error("missing LANCEDB_PLUGIN_ROOT or LANCEDB_DB_PATH");
}

const require = createRequire(import.meta.url);
const modulePath = require.resolve("@lancedb/lancedb", { paths: [pluginRoot] });
const lancedb = await import(pathToFileURL(modulePath).href);
const db = await lancedb.connect(dbPath);
const table = await db.openTable(tableName);
const rows = await table.query().toArray();
const compact = rows.map((row) => ({
  text: row.text,
  category: row.category,
  scope: row.scope,
  timestamp: Number(row.timestamp),
  metadata: row.metadata,
}));
process.stdout.write(JSON.stringify(compact));
"#;

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

pub fn load_from_memory_lancedb<P: AsRef<Path>>(db_path: P) -> Result<Vec<CorpusEntry>> {
    let db_path = db_path.as_ref();
    if is_json_export_path(db_path) {
        return load_memory_export_file(db_path);
    }

    match load_lancedb_rows_via_node(db_path) {
        Ok(entries) if !entries.is_empty() => Ok(dedup_entries(entries)),
        Ok(_) => {
            if let Some(backup) = find_nearest_memory_backup(db_path)? {
                load_memory_export_file(&backup)
            } else {
                Ok(Vec::new())
            }
        }
        Err(node_error) => {
            if let Some(backup) = find_nearest_memory_backup(db_path)? {
                let entries = load_memory_export_file(&backup).with_context(|| {
                    format!(
                        "failed to read LanceDB backup after live read failed for {}",
                        db_path.display()
                    )
                })?;
                return Ok(entries);
            }
            Err(node_error).with_context(|| {
                format!(
                    "failed to read memories from LanceDB source {}",
                    db_path.display()
                )
            })
        }
    }
}

pub fn load_from_session_logs<P: AsRef<Path>>(sessions_dir: P) -> Result<Vec<CorpusEntry>> {
    let sessions_dir = sessions_dir.as_ref();
    let mut files = Vec::new();
    collect_jsonl_files(sessions_dir, &mut files)
        .with_context(|| format!("failed to scan {}", sessions_dir.display()))?;
    files.sort();

    let mut entries = Vec::new();
    for file in files {
        entries.extend(load_session_log_file(&file)?);
    }
    Ok(dedup_entries(entries))
}

pub fn load_from_git_commits(repo_paths: &[PathBuf]) -> Result<Vec<CorpusEntry>> {
    let mut entries = Vec::new();
    for repo_path in repo_paths {
        let repo_name = repo_path
            .file_name()
            .and_then(OsStr::to_str)
            .unwrap_or("repo")
            .to_string();
        let output = Command::new("git")
            .arg("-C")
            .arg(repo_path)
            .arg("log")
            .arg("--format=%ct%x1f%s%x1f%b%x1e")
            .output()
            .with_context(|| format!("failed to run git log in {}", repo_path.display()))?;
        if !output.status.success() {
            bail!(
                "git log failed in {}: {}",
                repo_path.display(),
                String::from_utf8_lossy(&output.stderr).trim()
            );
        }

        let stdout = String::from_utf8(output.stdout)
            .with_context(|| format!("git log output was not UTF-8 for {}", repo_path.display()))?;
        for record in stdout.split('\u{1e}') {
            let record = record.trim();
            if record.is_empty() {
                continue;
            }
            let mut parts = record.split('\u{1f}');
            let timestamp = parts
                .next()
                .and_then(|value| value.trim().parse::<i64>().ok())
                .unwrap_or_default();
            let subject = parts.next().unwrap_or_default().trim();
            let body = parts.next().unwrap_or_default().trim();
            let text = match (subject.is_empty(), body.is_empty()) {
                (true, true) => continue,
                (_, true) => subject.to_string(),
                (true, false) => body.to_string(),
                (false, false) => format!("{subject}\n\n{body}"),
            };
            entries.push(CorpusEntry {
                text,
                category: format!("git:{repo_name}"),
                timestamp,
                source: format!("git:{repo_name}"),
            });
        }
    }
    Ok(dedup_entries(entries))
}

pub fn merge_corpus_sources(sources: &[Vec<CorpusEntry>]) -> Vec<CorpusEntry> {
    let mut merged = Vec::new();
    for source in sources {
        merged.extend_from_slice(source);
    }
    dedup_entries(merged)
}

pub fn dedup_entries(entries: Vec<CorpusEntry>) -> Vec<CorpusEntry> {
    let mut deduped = BTreeMap::<String, CorpusEntry>::new();
    for entry in entries {
        if entry.text.trim().is_empty() {
            continue;
        }
        let key = normalized_text_key(&entry.text);
        match deduped.get_mut(&key) {
            Some(existing) => merge_duplicate_entry(existing, entry),
            None => {
                deduped.insert(key, entry);
            }
        }
    }

    let mut entries = deduped.into_values().collect::<Vec<_>>();
    entries.sort_by(|left, right| {
        left.timestamp
            .cmp(&right.timestamp)
            .then_with(|| left.source.cmp(&right.source))
            .then_with(|| left.text.cmp(&right.text))
    });
    entries
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

fn load_memory_export_file(path: &Path) -> Result<Vec<CorpusEntry>> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("failed to open memory export {}", path.display()))?;
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
        let row = serde_json::from_str::<MemoryExportRow>(&line).with_context(|| {
            format!(
                "failed to parse memory export line {} from {}",
                line_index + 1,
                path.display()
            )
        })?;
        entries.push(memory_row_to_corpus_entry(row));
    }
    Ok(dedup_entries(entries))
}

fn memory_row_to_corpus_entry(row: MemoryExportRow) -> CorpusEntry {
    let metadata_value = row
        .metadata
        .as_deref()
        .and_then(|raw| serde_json::from_str::<Value>(raw).ok());
    let abstract_text = metadata_value
        .as_ref()
        .and_then(|value| value.get("l0_abstract"))
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or("");
    let content_text = metadata_value
        .as_ref()
        .and_then(|value| value.get("l2_content"))
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or("");
    let category = metadata_value
        .as_ref()
        .and_then(|value| value.get("memory_category"))
        .and_then(Value::as_str)
        .unwrap_or(row.category.as_str())
        .to_string();
    let source = metadata_value
        .as_ref()
        .and_then(|value| value.get("source_session"))
        .and_then(Value::as_str)
        .map(|session| format!("memory:{session}"))
        .or_else(|| row.scope.as_ref().map(|scope| format!("memory:{scope}")))
        .unwrap_or_else(|| "memory:global".to_string());
    let text = if !content_text.is_empty() && content_text != abstract_text {
        if abstract_text.is_empty() {
            content_text.to_string()
        } else {
            format!("{abstract_text}\n\n{content_text}")
        }
    } else if !abstract_text.is_empty() {
        abstract_text.to_string()
    } else {
        row.text
    };

    CorpusEntry {
        text,
        category,
        timestamp: row.timestamp.unwrap_or_default(),
        source,
    }
}

fn load_lancedb_rows_via_node(path: &Path) -> Result<Vec<CorpusEntry>> {
    let plugin_root = detect_lancedb_plugin_root()?;
    let (db_path, table_name) = resolve_lancedb_path(path)?;
    let output = Command::new("node")
        .arg("--input-type=module")
        .arg("-e")
        .arg(LANCEDB_READER_SCRIPT)
        .env("LANCEDB_PLUGIN_ROOT", plugin_root)
        .env("LANCEDB_DB_PATH", &db_path)
        .env("LANCEDB_TABLE_NAME", table_name)
        .output()
        .with_context(|| "failed to launch node for LanceDB reader")?;
    if !output.status.success() {
        return Err(anyhow!(
            "node LanceDB reader failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }

    let rows = serde_json::from_slice::<Vec<MemoryExportRow>>(&output.stdout)
        .context("failed to parse LanceDB reader output")?;
    Ok(dedup_entries(
        rows.into_iter().map(memory_row_to_corpus_entry).collect(),
    ))
}

fn detect_lancedb_plugin_root() -> Result<String> {
    if let Ok(path) = std::env::var("TURBOCALM_LANCEDB_PLUGIN_ROOT") {
        if Path::new(&path).exists() {
            return Ok(path);
        }
    }

    let home = std::env::var("HOME").context("HOME is not set")?;
    let candidates = [
        PathBuf::from(&home)
            .join(".openclaw")
            .join("extensions")
            .join("memory-lancedb-pro"),
        PathBuf::from(&home)
            .join("openclaw_src")
            .join("extensions")
            .join("memory-lancedb-pro"),
    ];

    for candidate in candidates {
        if candidate.exists() {
            return Ok(candidate.display().to_string());
        }
    }

    bail!("could not locate memory-lancedb-pro plugin root; set TURBOCALM_LANCEDB_PLUGIN_ROOT")
}

fn resolve_lancedb_path(path: &Path) -> Result<(String, String)> {
    if path
        .extension()
        .and_then(OsStr::to_str)
        .is_some_and(|extension| extension.eq_ignore_ascii_case("lance"))
    {
        let table_name = path
            .file_stem()
            .and_then(OsStr::to_str)
            .context("failed to derive LanceDB table name from .lance path")?;
        let db_dir = path
            .parent()
            .context("LanceDB .lance path must have a parent directory")?;
        return Ok((db_dir.display().to_string(), table_name.to_string()));
    }

    Ok((path.display().to_string(), "memories".to_string()))
}

fn find_nearest_memory_backup(path: &Path) -> Result<Option<PathBuf>> {
    let candidate_roots = if path.is_dir() {
        vec![path.to_path_buf(), path.join("..")]
    } else if path
        .extension()
        .and_then(OsStr::to_str)
        .is_some_and(|extension| extension.eq_ignore_ascii_case("lance"))
    {
        let parent = path
            .parent()
            .context("expected .lance path to have a parent directory")?;
        vec![parent.to_path_buf(), parent.join("..")]
    } else {
        vec![path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf()]
    };

    let mut backups = Vec::new();
    for root in candidate_roots {
        let backup_dir = root.join("backups");
        if !backup_dir.exists() {
            continue;
        }
        for entry in std::fs::read_dir(&backup_dir)
            .with_context(|| format!("failed to read {}", backup_dir.display()))?
        {
            let entry =
                entry.with_context(|| format!("failed to read {}", backup_dir.display()))?;
            let entry_path = entry.path();
            let is_backup = entry_path
                .file_name()
                .and_then(OsStr::to_str)
                .is_some_and(|name| name.starts_with("memory-backup-") && name.ends_with(".jsonl"));
            if is_backup {
                backups.push(entry_path);
            }
        }
    }

    backups.sort();
    Ok(backups.pop())
}

fn is_json_export_path(path: &Path) -> bool {
    path.extension()
        .and_then(OsStr::to_str)
        .is_some_and(|extension| matches!(extension, "jsonl" | "json"))
}

fn collect_jsonl_files(path: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    if path.is_file() {
        if path
            .extension()
            .and_then(OsStr::to_str)
            .is_some_and(|extension| extension.eq_ignore_ascii_case("jsonl"))
        {
            files.push(path.to_path_buf());
        }
        return Ok(());
    }

    if !path.is_dir() {
        bail!("{} is not a file or directory", path.display());
    }

    for entry in
        std::fs::read_dir(path).with_context(|| format!("failed to read {}", path.display()))?
    {
        let entry = entry.with_context(|| format!("failed to read {}", path.display()))?;
        let child = entry.path();
        if child.is_dir() {
            collect_jsonl_files(&child, files)?;
        } else if child
            .extension()
            .and_then(OsStr::to_str)
            .is_some_and(|extension| extension.eq_ignore_ascii_case("jsonl"))
        {
            files.push(child);
        }
    }
    Ok(())
}

fn load_session_log_file(path: &Path) -> Result<Vec<CorpusEntry>> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("failed to open session log {}", path.display()))?;
    let reader = BufReader::new(file);
    let session_source = format!(
        "session:{}",
        path.file_stem()
            .and_then(OsStr::to_str)
            .unwrap_or("unknown")
    );
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
        let value = serde_json::from_str::<Value>(&line).with_context(|| {
            format!(
                "failed to parse session log line {} from {}",
                line_index + 1,
                path.display()
            )
        })?;
        let Some(role) = value
            .get("message")
            .and_then(|message| message.get("role"))
            .and_then(Value::as_str)
        else {
            continue;
        };

        let text = extract_session_text(&value);
        if text.trim().is_empty() {
            continue;
        }

        entries.push(CorpusEntry {
            text,
            category: format!("session:{role}"),
            timestamp: parse_timestamp_value(value.get("timestamp")).unwrap_or_default(),
            source: session_source.clone(),
        });
    }

    Ok(entries)
}

fn extract_session_text(value: &Value) -> String {
    let Some(content) = value
        .get("message")
        .and_then(|message| message.get("content"))
    else {
        return String::new();
    };

    if let Some(text) = content.as_str() {
        return text.trim().to_string();
    }

    let Some(blocks) = content.as_array() else {
        return String::new();
    };

    blocks
        .iter()
        .filter_map(|block| {
            if block.get("type").and_then(Value::as_str) != Some("text") {
                return None;
            }
            block.get("text").and_then(Value::as_str).map(str::trim)
        })
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn parse_timestamp_value(value: Option<&Value>) -> Option<i64> {
    let value = value?;
    if let Some(timestamp) = value.as_i64() {
        return Some(timestamp);
    }
    let text = value.as_str()?;
    if let Ok(timestamp) = text.parse::<i64>() {
        return Some(timestamp);
    }
    DateTime::parse_from_rfc3339(text)
        .ok()
        .map(|timestamp| timestamp.timestamp_millis())
}

fn normalized_text_key(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

fn merge_duplicate_entry(existing: &mut CorpusEntry, candidate: CorpusEntry) {
    if candidate.timestamp != 0
        && (existing.timestamp == 0 || candidate.timestamp < existing.timestamp)
    {
        existing.timestamp = candidate.timestamp;
    }
    if existing.category.starts_with("session:") && !candidate.category.starts_with("session:") {
        existing.category = candidate.category;
    }
    if existing.source.starts_with("session:") && !candidate.source.starts_with("session:") {
        existing.source = candidate.source;
    }
    if candidate.text.len() > existing.text.len() {
        existing.text = candidate.text;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

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

    #[test]
    fn session_logs_are_parsed_into_entries() -> Result<()> {
        let temp_dir = temp_dir("sessions");
        let log_path = temp_dir.join("sample.jsonl");
        std::fs::write(
            &log_path,
            concat!(
                "{\"type\":\"message\",\"timestamp\":\"2026-03-29T13:00:35.265Z\",\"message\":{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"hello from user\"}]}}\n",
                "{\"type\":\"message\",\"timestamp\":\"2026-03-29T13:00:36.265Z\",\"message\":{\"role\":\"assistant\",\"content\":[{\"type\":\"text\",\"text\":\"hello from assistant\"}]}}\n"
            ),
        )?;

        let entries = load_from_session_logs(&temp_dir)?;
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].category, "session:user");
        assert!(entries[1].text.contains("assistant"));
        std::fs::remove_dir_all(temp_dir)?;
        Ok(())
    }

    #[test]
    fn memory_exports_prefer_rich_metadata_text() -> Result<()> {
        let temp_dir = temp_dir("memory");
        let export_path = temp_dir.join("memory-backup.jsonl");
        std::fs::write(
            &export_path,
            concat!(
                "{\"text\":\"short abstract\",\"category\":\"fact\",\"scope\":\"global\",\"timestamp\":1,\"metadata\":\"{\\\"l0_abstract\\\":\\\"short abstract\\\",\\\"l2_content\\\":\\\"full content\\\",\\\"memory_category\\\":\\\"preferences\\\"}\"}\n"
            ),
        )?;

        let entries = load_from_memory_lancedb(&export_path)?;
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].category, "preferences");
        assert!(entries[0].text.contains("full content"));
        std::fs::remove_dir_all(temp_dir)?;
        Ok(())
    }

    #[test]
    fn git_commits_are_loaded() -> Result<()> {
        let temp_dir = temp_dir("git");
        Command::new("git")
            .arg("init")
            .arg(&temp_dir)
            .output()
            .context("failed to init git repo")?;
        Command::new("git")
            .arg("-C")
            .arg(&temp_dir)
            .arg("config")
            .arg("user.name")
            .arg("TurboCALM")
            .output()?;
        Command::new("git")
            .arg("-C")
            .arg(&temp_dir)
            .arg("config")
            .arg("user.email")
            .arg("turbocalm@example.com")
            .output()?;
        std::fs::write(temp_dir.join("README.md"), "hello")?;
        Command::new("git")
            .arg("-C")
            .arg(&temp_dir)
            .arg("add")
            .arg(".")
            .output()?;
        Command::new("git")
            .arg("-C")
            .arg(&temp_dir)
            .arg("commit")
            .arg("-m")
            .arg("seed commit")
            .output()?;

        let entries = load_from_git_commits(&[temp_dir.clone()])?;
        assert_eq!(entries.len(), 1);
        assert!(entries[0].text.contains("seed commit"));
        std::fs::remove_dir_all(temp_dir)?;
        Ok(())
    }

    #[test]
    fn merge_sources_dedups_by_normalized_text() {
        let entries = merge_corpus_sources(&[
            vec![CorpusEntry {
                text: "Hello   world".to_string(),
                category: "a".to_string(),
                timestamp: 2,
                source: "x".to_string(),
            }],
            vec![CorpusEntry {
                text: "hello world".to_string(),
                category: "b".to_string(),
                timestamp: 1,
                source: "y".to_string(),
            }],
        ]);

        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].timestamp, 1);
    }

    fn temp_dir(label: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "turbocalm-train-corpus-{label}-{}-{unique}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path).unwrap();
        path
    }
}
