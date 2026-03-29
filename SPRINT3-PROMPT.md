# Sprint 3: Evaluation + HTTP Server + Integration

READ FIRST:
- crates/turbocalm-train/src/ (everything from Sprint 2)
- PLAN-self-training-embeddings.md (the full unified plan)
- crates/turbocalm-models/src/calm/autoencoder.rs
- crates/turbocalm-triumvirate/src/lib.rs (EmbeddingEngine trait)

## BUILD:

### 1. eval.rs — Evaluation harness
- EvalPair { query: String, relevant_ids: Vec<String> }
- EvalCorpus { pairs: Vec<EvalPair>, documents: Vec<(String, String)> } // (id, text)
- run_eval(model, corpus) → EvalMetrics { recall_at_5: f32, mrr: f32, avg_cosine_similar: f32, avg_cosine_dissimilar: f32 }
- Embed all docs, for each query find top-K by cosine sim, compute recall@5 and MRR
- Print results as formatted table

### 2. server.rs — HTTP embedding server
- OpenAI-compatible /v1/embeddings endpoint
- POST { input: "text" | ["text1", "text2"], model: "turbocalm-local" }
- Response: { data: [{ embedding: [...], index: 0 }], model: "turbocalm-local", usage: { prompt_tokens: N, total_tokens: N } }
- Use axum or tiny_http (add to Cargo.toml as needed)
- Load trained checkpoint on startup, fall back to random weights
- --port flag (default 11435)
- --pooled mode (centroid, default) and --chunked mode (per-patch vectors)

### 3. CLI commands (update main.rs)
- `turbocalm-train train --corpus <path> --epochs 20 --lr 1e-4 --checkpoint-dir ~/.turbocalm/trained/`
- `turbocalm-train eval --corpus <path> --checkpoint <path>`
- `turbocalm-train serve --port 11435 --checkpoint <path>`
- `turbocalm-train corpus build --sources <jsonl-paths> --output corpus.jsonl`
- `turbocalm-train checkpoints list`

### 4. Integration with turbocalm-triumvirate
- Modify EmbeddingEngine in turbocalm-triumvirate to check for trained weights at ~/.turbocalm/trained/latest.safetensors
- If found, load trained weights instead of HF/zeroed weights
- Add config option: prefer_trained: bool (default true)

### 5. Tests
- eval test: synthetic corpus, verify recall@5 > 0 after training
- server test: spawn server, POST /v1/embeddings, verify 128-dim response
- Integration test: train → save → load in EmbeddingEngine → verify non-zero cosine similarity between similar texts

CONSTRAINTS:
- Server must handle concurrent requests safely (Arc<Model>)
- Checkpoint loading must be atomic (load to temp, verify, rename)
- All tests must pass: cargo test -p turbocalm-train

When completely finished, run: openclaw system event --text "Done: Sprint 3 complete — eval, HTTP server, and integration built" --mode now
