# Sprint 4-5: Online Learning + Calibration + Polish

READ FIRST:
- Everything in crates/turbocalm-train/src/
- PLAN-self-training-embeddings.md
- Triumvirate thresholds (for context only — don't modify Triumvirate repo):
  - SEMANTIC_DUPLICATE_THRESHOLD = 0.24 (ingestion.rs)
  - CLUSTER_SIMILARITY_THRESHOLD = 0.08 (consolidation.rs)
  - SEMANTIC_CONVERGENCE_THRESHOLD = 0.11 (distillation.rs)

## BUILD:

### 1. online.rs — Incremental learning
- OnlineLearner struct with internal buffer
- add_text(text: &str, category: &str) — buffer new texts
- When buffer reaches threshold (default 50), auto-generate pairs and train 1-3 mini-epochs
- Save updated checkpoint after each online training batch
- Thread-safe: can be called from HTTP server endpoint

### 2. calibrate.rs — Threshold calibration
- Run trained model on labeled pairs at 4 similarity tiers (exact, near-dup, related, unrelated)
- Compute similarity distributions per tier
- Recommend thresholds: dedup, cluster, convergence
- Output calibration.toml with recommended values
- CLI: `turbocalm-train calibrate --checkpoint <path> --corpus <eval-corpus>`

### 3. corpus.rs enhancements — Real data sources
- load_from_memory_lancedb(db_path) — read memories from LanceDB pro database
- load_from_session_logs(sessions_dir) — parse OpenClaw .jsonl session logs
- load_from_git_commits(repo_paths) — extract commit messages via git log
- Merge multiple sources into unified corpus with dedup

### 4. CLI enhancements
- `turbocalm-train train --online` — start online learner daemon
- `turbocalm-train calibrate --checkpoint <path>`
- `turbocalm-train corpus build --memory-db <path> --sessions <dir> --git-repos <paths> --output corpus.jsonl`

### 5. README.md update
- Document all CLI commands
- Quick start guide
- Architecture diagram
- Configuration reference

### 6. Final integration test
- Build corpus from synthetic data
- Train 10 epochs
- Run eval, verify recall@5 > 0.5
- Save checkpoint
- Start HTTP server with checkpoint
- POST /v1/embeddings, verify response
- Run calibrate, verify it produces thresholds
- Online learner: add 50 texts, verify auto-trains

Run all tests: cargo test -p turbocalm-train
Verify clean build: cargo build --release -p turbocalm-train

When completely finished, run: openclaw system event --text "Done: Sprint 4-5 complete — online learning, calibration, and polish done. turbocalm-train is feature complete." --mode now
