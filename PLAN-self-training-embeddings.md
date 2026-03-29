# Plan: Self-Training CALM Autoencoder — Unified for Triumvirate + OpenClaw Memory

## Vision

candle-turbocalm becomes a **self-improving embedding backbone** that serves two systems:

1. **Triumvirate/Acolyte** — Axiom embeddings, energy scoring, deduplication, consolidation, and inter-council knowledge distillation
2. **OpenClaw/Caelator** — Long-term memory storage, recall, and semantic search via memory-lancedb-pro

Both systems already consume CALM embeddings through the same interface (`EmbeddingEngine` trait / OpenAI-compatible HTTP). The missing piece is *trained weights* — and the ability to continuously improve those weights from both systems' data.

## Architecture

```
                    ┌──────────────────────────────────────┐
                    │         turbocalm-train               │
                    │   (new crate — training engine)       │
                    │                                       │
                    │  Corpus ← Acolyte axioms              │
                    │         ← OpenClaw memories            │
                    │         ← Git commits                  │
                    │         ← Session logs                 │
                    │                                       │
                    │  Contrastive Training (NT-Xent)       │
                    │  on Apple Silicon Metal                │
                    │                                       │
                    │  Trained weights → safetensors         │
                    └────────────┬─────────────────────────┘
                                 │
                    ┌────────────▼─────────────────────────┐
                    │      CALM Autoencoder (75M params)     │
                    │      128-dim latent embeddings         │
                    │      Metal GPU inference               │
                    └────────┬──────────────┬──────────────┘
                             │              │
              ┌──────────────▼──┐    ┌──────▼───────────────┐
              │   Triumvirate    │    │   OpenClaw Memory    │
              │                  │    │                      │
              │  EmbeddingEngine │    │  /v1/embeddings HTTP │
              │  trait (Rust)    │    │  (OpenAI-compat)     │
              │                  │    │                      │
              │  • Axiom ingest  │    │  • memory_store      │
              │  • Dedup (0.24)  │    │  • memory_recall     │
              │  • Cluster(0.08) │    │  • auto-capture      │
              │  • Distill(0.11) │    │  • auto-recall       │
              │  • Energy/VQS    │    │                      │
              │                  │    │  memory-lancedb-pro  │
              │  acolyte-embed   │    │  → baseURL:localhost  │
              └──────────────────┘    └──────────────────────┘
```

## What Gets Built

### New Crate: `turbocalm-train`

```
crates/turbocalm-train/
├── Cargo.toml
├── src/
│   ├── lib.rs              # Public API
│   ├── corpus.rs           # Unified corpus builder
│   ├── pairs.rs            # Contrastive pair generation
│   ├── loss.rs             # NT-Xent contrastive loss
│   ├── trainer.rs          # Training loop (Metal autograd)
│   ├── checkpoint.rs       # Versioned weight management
│   ├── online.rs           # Incremental learning
│   ├── eval.rs             # Evaluation (wraps acolyte-embeddings eval harness)
│   └── server.rs           # HTTP /v1/embeddings server
```

### Modified Existing Crates

- **turbocalm-models** — Make autoencoder weights `Var`-backed for training mode (currently `Tensor`-only for inference)
- **turbocalm-triumvirate** — `EmbeddingEngine` and `EnergyScorer` load trained weights when available, fall back to HF/zeroed
- **turbocalm-cli** — New subcommands: `train`, `corpus`, `eval`, `serve-embeddings`

## Dual-Source Corpus

The key insight: both systems produce semantically structured text that can train the same model.

### From Triumvirate/Acolyte

| Source | What | Category Signal |
|---|---|---|
| Axioms | Ingested knowledge with `KnowledgeCategory` tags | Category + domain tags |
| Council memories | Task-specific learnings with quality scores | Quality score + council ID |
| Distilled principles | High-confidence cross-council patterns | Always high-value positives |
| Eval corpus | 50+ calibrated query→axiom pairs with similarity tiers | Ground truth labels |

### From OpenClaw/Caelator

| Source | What | Category Signal |
|---|---|---|
| memory_store entries | Preferences, facts, decisions, entities | LanceDB `category` field |
| Session logs | Conversation turns (.jsonl) | Temporal proximity |
| Daily notes | memory/YYYY-MM-DD.md files | Date proximity |
| Git commits | Commit messages from indexed repos | Repo + file proximity |

### Pair Generation Strategy

**Positive pairs (should be similar):**
- Same axiom category (e.g., two `ArchitecturalPrinciple` axioms)
- Same memory category (e.g., two `preference` memories)
- Axiom ↔ its distilled principle (semantic entailment)
- Memory ↔ query that recalled it (retrieval feedback)
- Eval corpus known-similar pairs (ground truth from calibration harness)
- Temporally adjacent session messages

**Hard negatives (should be dissimilar):**
- Cross-category pairs (architecture axiom vs. medical memory)
- Cross-domain pairs (Triumvirate governance vs. BE SATAS clinical)
- Eval corpus known-dissimilar pairs
- Random sampling from distant time windows

## Training Loop

```rust
pub struct TrainingConfig {
    pub batch_size: usize,          // 32
    pub learning_rate: f64,         // 1e-4
    pub weight_decay: f64,          // 0.01
    pub temperature: f64,           // 0.07
    pub max_epochs: usize,          // 20
    pub eval_interval: usize,       // Every 2 epochs
    pub patience: usize,            // 3 (early stopping)
    pub checkpoint_dir: PathBuf,    // ~/.turbocalm/trained/
    pub min_corpus_size: usize,     // 200 pairs minimum
}
```

**Key details:**
- Uses Candle `Var` for trainable weights + `AdamW` optimizer
- NT-Xent loss with in-batch negatives (no explicit negative mining needed)
- Gradient clipping at 1.0
- Eval harness runs every 2 epochs using Acolyte's existing `EvalCorpus` format
- Early stopping on recall@5 plateau
- Checkpoints saved as safetensors with monotonic version numbers

## Threshold Recalibration

This is critical. Acolyte's pipeline has four hardcoded thresholds calibrated for the current (zero-weight) CALM model:

| Constant | Current | Location |
|---|---|---|
| `SEMANTIC_DUPLICATE_THRESHOLD` | 0.24 | `ingestion.rs` |
| `CLUSTER_SIMILARITY_THRESHOLD` | 0.08 | `consolidation.rs` |
| `SEMANTIC_CONVERGENCE_THRESHOLD` | 0.11 | `distillation.rs` |
| `DeduplicationConfig.similarity_threshold` | 0.08 | `ingestion.rs` |

After training, these thresholds will shift because the embedding space changes. The plan:

1. After each training run, auto-run Acolyte's calibration harness (`acolyte-embeddings/src/calibrate.rs`)
2. The harness produces recommended thresholds from the eval corpus similarity distributions
3. Output a `calibration.toml` that Acolyte can load at runtime instead of hardcoded constants
4. Fallback: if no calibration file exists, use current hardcoded values

This means trained weights and their calibrated thresholds are always paired — you never get a mismatch.

## Evaluation: Unified Metrics

Both systems need to validate that trained embeddings are actually better than untrained. We reuse and extend Acolyte's eval harness:

**Acolyte metrics (existing):**
- Precision@K, Recall@K, MRR on axiom retrieval
- Similarity distribution stats (similar_mean, similar_std, dissimilar_mean, dissimilar_std)
- Suggested thresholds from distribution analysis

**Memory metrics (new, same format):**
- Precision@K, Recall@K, MRR on memory recall
- Built from memory_store/memory_recall pairs (retrieval feedback)

**Combined score:**
- Weighted geometric mean of Acolyte MRR + Memory MRR
- Training only promotes weights when combined score improves over previous best
- This prevents overfitting to one system at the expense of the other

## HTTP Embedding Server

Serves trained CALM embeddings in OpenAI `/v1/embeddings` format:

```
POST http://localhost:11435/v1/embeddings
Content-Type: application/json

{
  "input": "text to embed",
  "model": "turbocalm-local"
}
```

**Two modes:**
- `--pooled` (default for memory-lancedb-pro): Returns single 128-dim centroid vector
- `--chunked` (for Acolyte ColBERT-style): Returns per-chunk vectors + centroid

memory-lancedb-pro config change:
```json
{
  "embedding": {
    "provider": "openai-compatible",
    "baseURL": "http://localhost:11435/v1",
    "model": "turbocalm-local",
    "dimensions": 128
  }
}
```

Acolyte already loads via the Rust `EmbeddingEngine` trait — it just needs to find the trained safetensors at the configured path.

## CLI Commands

```bash
# Build corpus from both systems
turbocalm corpus build \
  --acolyte-axioms /path/to/triumvirate/acolyte/data \
  --openclaw-memories ~/.openclaw/memory/lancedb-pro \
  --sessions ~/.openclaw/agents/main/sessions \
  --git-repos ~/Desktop/GitHub/Triumvirate,~/Desktop/GitHub/be.satas \
  --output corpus.jsonl

# Train
turbocalm train \
  --corpus corpus.jsonl \
  --epochs 20 \
  --eval-corpus /path/to/acolyte/eval/corpus.json \
  --calibrate  # auto-recalibrate Acolyte thresholds after training

# Evaluate against both systems
turbocalm eval \
  --corpus corpus.jsonl \
  --acolyte-eval /path/to/eval/corpus.json \
  --memory-eval  # auto-build from LanceDB access logs

# Serve for memory-lancedb-pro
turbocalm serve-embeddings --port 11435

# Online incremental training
turbocalm train --online --buffer-size 50

# Show training status and checkpoint history
turbocalm train status
```

## Implementation Sprints

### Sprint 1: Training Foundation (Day 1-2)
1. Create `turbocalm-train` crate
2. Modify `turbocalm-models` autoencoder to support `Var`-backed weights (training mode)
3. Implement NT-Xent loss (`loss.rs`)
4. Implement training loop with AdamW (`trainer.rs`)
5. Implement checkpoint save/load as safetensors (`checkpoint.rs`)
6. Test: train on synthetic pairs, verify loss decreases, weights roundtrip

### Sprint 2: Unified Corpus & Pairs (Day 2-3)
7. Implement corpus builder with Acolyte axiom + OpenClaw memory ingestors (`corpus.rs`)
8. Implement pair generation: category-based, temporal, eval-ground-truth, cross-domain negatives (`pairs.rs`)
9. Add `corpus build` CLI command
10. Test: build corpus from real Triumvirate + OpenClaw data

### Sprint 3: Evaluation & Calibration (Day 3-4)
11. Wrap Acolyte's eval harness for use from `turbocalm-train` (`eval.rs`)
12. Add memory recall evaluation (build eval pairs from LanceDB access logs)
13. Add combined scoring (geometric mean of Acolyte MRR + Memory MRR)
14. Add auto-calibration: run calibration harness → emit `calibration.toml`
15. Modify Acolyte threshold constants to load from calibration file when present

### Sprint 4: HTTP Server & Integration (Day 4-5)
16. Implement `/v1/embeddings` HTTP server (`server.rs`) using `axum` or `tiny_http`
17. Support both pooled (centroid) and chunked response modes
18. Add `serve-embeddings` CLI command
19. Modify `turbocalm-triumvirate` `EmbeddingEngine` to prefer trained weights
20. Test: memory-lancedb-pro → localhost → CALM → verify recall quality

### Sprint 5: Online Learning & Polish (Day 5-6)
21. Implement buffered online learning (`online.rs`)
22. Add combined eval gate: only promote weights when both systems improve
23. Wire up OpenClaw cron for periodic corpus rebuild + retrain
24. Update all documentation (turbocalm README, Acolyte architecture docs)
25. Add `train --online` and `train status` CLI commands

## Cold Start Strategy

1. **Now**: Gemini embeddings for OpenClaw memory, zeroed weights for Acolyte
2. **~200 memories + existing eval corpus**: First training batch, run combined eval
3. **If combined MRR improves**: Switch both systems to trained weights + recalibrated thresholds
4. **Ongoing**: Online learning ingests new data from both systems, periodic retraining

## Dependencies

New workspace deps (all lightweight):
- `axum` or `tiny_http` — HTTP server for embedding endpoint
- No other new deps — Candle autograd + safetensors + tokenizers already in workspace

## Success Criteria

1. Trained weights produce higher combined MRR than untrained on both eval corpora
2. Acolyte thresholds auto-recalibrate and pipeline tests pass
3. memory-lancedb-pro recall quality matches or exceeds Gemini embeddings
4. HTTP server latency < 10ms per embedding (Metal inference)
5. Online learning improves quality over time without manual intervention
6. Single `turbocalm train` command trains one model that serves both systems
