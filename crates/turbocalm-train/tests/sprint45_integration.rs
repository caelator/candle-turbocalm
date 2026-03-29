use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use bytes::Bytes;
use candle_core::Device;
use http::{Method, Request};
use http_body_util::{BodyExt, Full};
use hyper::client::conn::http1::handshake;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper_util::rt::TokioIo;
use serde_json::json;
use tokio::io::duplex;
use turbocalm_models::CalmAutoencoderConfig;
use turbocalm_train::server::{ServerState, DEFAULT_MODEL_NAME};
use turbocalm_train::{
    build_pairs_from_entries, load_from_git_commits, load_from_jsonl, load_from_memory_lancedb,
    load_from_session_logs, merge_corpus_sources, run_calibration, run_eval, save_calibration_toml,
    save_to_jsonl, CalibrationCorpus, CalibrationPair, EmbeddingMode, EmbeddingModel, EvalCorpus,
    EvalPair, OnlineLearner, OnlineLearningConfig, SimilarityTier, Trainer, TrainingConfig,
};

#[tokio::test]
async fn sprint45_end_to_end_pipeline() -> Result<()> {
    let temp_dir = TempDirGuard::new("sprint45")?;
    let memory_export = write_memory_export(temp_dir.path())?;
    let sessions_dir = write_session_logs(temp_dir.path())?;
    let git_repos = write_git_repos(temp_dir.path())?;

    let entries = merge_corpus_sources(&[
        load_from_memory_lancedb(&memory_export)?,
        load_from_session_logs(&sessions_dir)?,
        load_from_git_commits(&git_repos)?,
    ]);
    assert!(
        entries.len() >= 25,
        "expected merged synthetic corpus entries"
    );

    let corpus_path = temp_dir.path().join("corpus.jsonl");
    save_to_jsonl(&entries, &corpus_path)?;
    let entries = load_from_jsonl(&corpus_path)?;
    let pairs = build_pairs_from_entries(&entries);
    assert!(
        pairs.pairs.len() >= 20,
        "expected training pairs from synthetic corpus"
    );

    let model_config = small_model_config(32);
    let checkpoint_dir = temp_dir.path().join("checkpoints");
    let training_config = TrainingConfig {
        batch_size: 8,
        lr: 0.02,
        weight_decay: 0.0,
        temperature: 0.07,
        max_epochs: 10,
        eval_interval: 1,
        patience: 10,
        checkpoint_dir: checkpoint_dir.clone(),
        min_corpus_size: 1,
    };

    let mut trainer = Trainer::new(model_config.clone(), training_config.clone(), Device::Cpu)?;
    let summary = trainer.train(&pairs)?;
    let checkpoint_path = summary
        .best_checkpoint
        .as_ref()
        .context("expected best checkpoint after training")?
        .path
        .clone();

    let model =
        EmbeddingModel::from_checkpoint(&checkpoint_path, model_config.clone(), Device::Cpu)?;
    let metrics = run_eval(&model, &synthetic_eval_corpus())?;
    assert!(
        metrics.recall_at_5 > 0.5,
        "expected recall@5 > 0.5, got {}",
        metrics.recall_at_5
    );

    let state = ServerState::new(model, EmbeddingMode::Pooled);
    let (client_io, server_io) = duplex(1 << 16);
    let server_state = state.clone();
    let server_task = tokio::spawn(async move {
        http1::Builder::new()
            .serve_connection(
                TokioIo::new(server_io),
                service_fn(move |request| {
                    turbocalm_train::server::handle_request_for_tests(request, server_state.clone())
                }),
            )
            .await
    });
    let (mut sender, connection) = handshake(TokioIo::new(client_io)).await?;
    let client = tokio::spawn(async move { connection.await });
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/embeddings")
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(serde_json::to_vec(&json!({
            "input": "calm inhale exhale breathing focus",
            "model": DEFAULT_MODEL_NAME,
        }))?)))?;
    let response = sender.send_request(request).await?;
    let body = response.into_body().collect().await?.to_bytes();
    let response: serde_json::Value = serde_json::from_slice(&body)?;
    let embedding = response["data"][0]["embedding"]
        .as_array()
        .context("expected pooled embedding array")?;
    assert_eq!(embedding.len(), model_config.latent_size);
    drop(sender);
    client.await??;
    server_task.await??;

    let calibration_corpus = synthetic_calibration_corpus();
    let model =
        EmbeddingModel::from_checkpoint(&checkpoint_path, model_config.clone(), Device::Cpu)?;
    let report = run_calibration(&model, &calibration_corpus)?;
    let calibration_path = temp_dir.path().join("calibration.toml");
    save_calibration_toml(&report, &calibration_path)?;
    let calibration_raw = std::fs::read_to_string(&calibration_path)?;
    assert!(calibration_raw.contains("[thresholds]"));
    assert!(report.recommended.dedup >= report.recommended.convergence);
    assert!(report.recommended.convergence >= report.recommended.cluster);

    let learner = OnlineLearner::from_checkpoint(
        &checkpoint_path,
        model_config,
        TrainingConfig {
            checkpoint_dir,
            min_corpus_size: 1,
            max_epochs: 3,
            eval_interval: 1,
            patience: 3,
            ..training_config
        },
        Device::Cpu,
        OnlineLearningConfig {
            buffer_size: 50,
            mini_epochs: 1,
        },
    )?;

    let themes = [
        "breathing",
        "architecture",
        "journaling",
        "debugging",
        "nutrition",
    ];
    let mut last_result = None;
    for index in 0..50 {
        let theme = themes[index % themes.len()];
        last_result = Some(learner.add_text(
            &format!("{theme} online sample {index} {}", theme_phrase(theme)),
            theme,
        )?);
    }

    let last_result = last_result.context("expected online learning result")?;
    assert!(last_result.trained);
    assert!(last_result.pair_count > 0);
    assert!(last_result.checkpoint.is_some());

    Ok(())
}

fn write_memory_export(root: &Path) -> Result<PathBuf> {
    let path = root.join("memory-backup.jsonl");
    let mut lines = Vec::new();
    for theme in [
        "breathing",
        "architecture",
        "journaling",
        "debugging",
        "nutrition",
    ] {
        for index in 0..2 {
            lines.push(serde_json::to_string(&json!({
                "text": format!("{theme} abstract {index}"),
                "category": "fact",
                "scope": "global",
                "timestamp": index + 1,
                "metadata": serde_json::to_string(&json!({
                    "l0_abstract": format!("{theme} abstract {index}"),
                    "l2_content": format!("{theme} detail {index} {}", theme_phrase(theme)),
                    "memory_category": theme,
                    "source_session": format!("memory-session-{theme}"),
                }))?,
            }))?);
        }
    }
    std::fs::write(&path, lines.join("\n"))?;
    Ok(path)
}

fn write_session_logs(root: &Path) -> Result<PathBuf> {
    let dir = root.join("sessions");
    std::fs::create_dir_all(&dir)?;
    for (offset, theme) in [
        "breathing",
        "architecture",
        "journaling",
        "debugging",
        "nutrition",
    ]
    .into_iter()
    .enumerate()
    {
        let path = dir.join(format!("{theme}.jsonl"));
        let lines = [
            json!({
                "type": "message",
                "timestamp": format!("2026-03-29T13:00:{:02}Z", offset),
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": format!("{theme} user {}", theme_phrase(theme))}]
                }
            }),
            json!({
                "type": "message",
                "timestamp": format!("2026-03-29T13:01:{:02}Z", offset),
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": format!("{theme} assistant {}", theme_phrase(theme))}]
                }
            }),
        ];
        std::fs::write(
            path,
            lines
                .into_iter()
                .map(|line| serde_json::to_string(&line))
                .collect::<std::result::Result<Vec<_>, _>>()?
                .join("\n"),
        )?;
    }
    Ok(dir)
}

fn write_git_repos(root: &Path) -> Result<Vec<PathBuf>> {
    let repos_dir = root.join("repos");
    std::fs::create_dir_all(&repos_dir)?;

    let mut repos = Vec::new();
    for theme in [
        "breathing",
        "architecture",
        "journaling",
        "debugging",
        "nutrition",
    ] {
        let repo = repos_dir.join(format!("{theme}-repo"));
        std::fs::create_dir_all(&repo)?;
        run_git(&repo, ["init"])?;
        run_git(&repo, ["config", "user.name", "TurboCALM"])?;
        run_git(&repo, ["config", "user.email", "turbocalm@example.com"])?;

        for commit_index in 0..2 {
            let file_path = repo.join(format!("entry-{commit_index}.txt"));
            std::fs::write(
                &file_path,
                format!("{theme} commit body {commit_index} {}", theme_phrase(theme)),
            )?;
            run_git(&repo, ["add", "."])?;
            run_git(
                &repo,
                [
                    "commit",
                    "-m",
                    &format!("{theme} commit {commit_index} {}", theme_phrase(theme)),
                ],
            )?;
        }

        repos.push(repo);
    }

    Ok(repos)
}

fn run_git<const N: usize>(repo: &Path, args: [&str; N]) -> Result<()> {
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(args)
        .output()
        .with_context(|| format!("failed to run git in {}", repo.display()))?;
    if !output.status.success() {
        anyhow::bail!(
            "git command failed in {}: {}",
            repo.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(())
}

fn synthetic_eval_corpus() -> EvalCorpus {
    EvalCorpus {
        pairs: vec![
            EvalPair {
                query: "calm inhale exhale breathing focus".to_string(),
                relevant_ids: vec!["doc-breathing".to_string()],
            },
            EvalPair {
                query: "event sourcing projections distributed systems".to_string(),
                relevant_ids: vec!["doc-architecture".to_string()],
            },
            EvalPair {
                query: "trace logs reproduce isolate failure".to_string(),
                relevant_ids: vec!["doc-debugging".to_string()],
            },
            EvalPair {
                query: "daily reflection gratitude planning notes".to_string(),
                relevant_ids: vec!["doc-journaling".to_string()],
            },
        ],
        documents: vec![
            (
                "doc-breathing".to_string(),
                format!("breathing document {}", theme_phrase("breathing")),
            ),
            (
                "doc-architecture".to_string(),
                format!("architecture document {}", theme_phrase("architecture")),
            ),
            (
                "doc-journaling".to_string(),
                format!("journaling document {}", theme_phrase("journaling")),
            ),
            (
                "doc-debugging".to_string(),
                format!("debugging document {}", theme_phrase("debugging")),
            ),
            (
                "doc-nutrition".to_string(),
                format!("nutrition document {}", theme_phrase("nutrition")),
            ),
            (
                "doc-distractor".to_string(),
                "astronomy telescopes galaxies orbit mechanics".to_string(),
            ),
        ],
    }
}

fn synthetic_calibration_corpus() -> CalibrationCorpus {
    CalibrationCorpus {
        pairs: vec![
            CalibrationPair {
                left: "steady breathing inhale exhale".to_string(),
                right: "steady breathing inhale exhale".to_string(),
                tier: SimilarityTier::Exact,
            },
            CalibrationPair {
                left: "steady breathing inhale exhale".to_string(),
                right: "calm breathing inhale and exhale practice".to_string(),
                tier: SimilarityTier::NearDup,
            },
            CalibrationPair {
                left: "event sourcing projections".to_string(),
                right: "distributed systems architecture".to_string(),
                tier: SimilarityTier::Related,
            },
            CalibrationPair {
                left: "event sourcing projections".to_string(),
                right: "garden compost tomatoes".to_string(),
                tier: SimilarityTier::Unrelated,
            },
        ],
    }
}

fn theme_phrase(theme: &str) -> &'static str {
    match theme {
        "breathing" => "steady breathing inhale exhale calm focus relaxation",
        "architecture" => "distributed systems event sourcing projection storage",
        "journaling" => "daily reflection gratitude notes planning review",
        "debugging" => "trace logs reproduction isolate failure fix",
        "nutrition" => "balanced meals vegetables protein hydration energy",
        _ => "general semantic training phrase",
    }
}

fn small_model_config(latent_size: usize) -> CalmAutoencoderConfig {
    CalmAutoencoderConfig {
        vocab_size: 512,
        hidden_size: 32,
        intermediate_size: 64,
        latent_size,
        num_encoder_layers: 2,
        num_decoder_layers: 2,
        patch_size: 4,
        max_position_embeddings: 128,
        tie_word_embeddings: true,
        ..Default::default()
    }
}

struct TempDirGuard {
    path: PathBuf,
}

impl TempDirGuard {
    fn new(label: &str) -> Result<Self> {
        let unique = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let path = std::env::temp_dir().join(format!(
            "turbocalm-train-{label}-{}-{unique}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path)?;
        Ok(Self { path })
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}
