# Candle-TurboCALM

Native Rust implementation of [CALM (Continuous Autoregressive Language Models)](https://arxiv.org/abs/2510.27688) powered by [Candle](https://github.com/huggingface/candle), with local training, checkpointing, evaluation, calibration, and OpenAI-compatible embedding serving.

## Features

- Pure Rust inference and training utilities
- CALM autoencoder checkpoint loading and local fine-tuning
- Multi-source corpus building from JSONL, OpenClaw session logs, LanceDB memory exports, and git history
- Online incremental learning with buffered auto-training
- Threshold calibration for dedup, clustering, and convergence
- OpenAI-compatible `/v1/embeddings` HTTP server
- Triumvirate embedding-engine compatibility via trained checkpoints

## Quick Start

Build the training crate:

```bash
cargo build --release -p turbocalm-train
```

Build a corpus from multiple sources:

```bash
cargo run -p turbocalm-train -- corpus build \
  --memory-db ~/.openclaw/memory/lancedb-pro/memories.lance \
  --sessions ~/.openclaw/agents/main/sessions \
  --git-repos ~/Desktop/GitHub/candle-turbocalm,~/Desktop/GitHub/Triumvirate \
  --output corpus.jsonl
```

Train a checkpoint offline:

```bash
cargo run -p turbocalm-train -- train \
  --corpus corpus.jsonl \
  --epochs 10 \
  --checkpoint-dir ./artifacts/checkpoints \
  --device cpu
```

Evaluate a checkpoint:

```bash
cargo run -p turbocalm-train -- eval \
  --corpus eval-corpus.json \
  --checkpoint ./artifacts/checkpoints/latest.safetensors \
  --device cpu
```

Serve pooled embeddings:

```bash
cargo run -p turbocalm-train -- serve \
  --checkpoint ./artifacts/checkpoints/latest.safetensors \
  --port 11435 \
  --device cpu
```

Calibrate thresholds from labeled pairs:

```bash
cargo run -p turbocalm-train -- calibrate \
  --checkpoint ./artifacts/checkpoints/latest.safetensors \
  --corpus calibration-corpus.json \
  --output ./artifacts/checkpoints/calibration.toml \
  --device cpu
```

Start the online learner daemon:

```bash
cargo run -p turbocalm-train -- train \
  --online \
  --checkpoint ./artifacts/checkpoints/latest.safetensors \
  --checkpoint-dir ./artifacts/checkpoints \
  --buffer-size 50 \
  --mini-epochs 2 \
  --port 11435 \
  --device cpu
```

## Architecture

```text
                    +------------------------------+
                    |        turbocalm-train       |
                    |------------------------------|
                    | corpus build                 |
                    | offline train                |
                    | online learner               |
                    | eval + calibrate             |
                    | /v1/embeddings server        |
                    +---------------+--------------+
                                    |
                           checkpoints (.safetensors)
                                    |
                    +---------------v--------------+
                    |     CALM autoencoder         |
                    |     128-dim latent space     |
                    +---------------+--------------+
                                    |
              +---------------------+----------------------+
              |                                            |
  +-----------v-----------+                    +-----------v-----------+
  | turbocalm-triumvirate |                    | OpenAI-compatible HTTP|
  | EmbeddingEngine       |                    | /v1/embeddings        |
  +-----------------------+                    +-----------------------+
```

## `turbocalm-train` CLI

### `train`

Offline training:

```bash
turbocalm-train train --corpus corpus.jsonl --epochs 10 --device cpu
```

Online mode:

```bash
turbocalm-train train --online --buffer-size 50 --mini-epochs 2 --port 11435 --device cpu
```

Key options:

- `--corpus <path>`: JSONL corpus for offline training
- `--checkpoint <path>`: load an existing checkpoint before serving or online training
- `--checkpoint-dir <path>`: directory for versioned checkpoints and `latest.safetensors`
- `--epochs <n>`: max offline epochs
- `--batch-size <n>`
- `--lr <f64>`
- `--weight-decay <f64>`
- `--temperature <f64>`
- `--eval-interval <n>`
- `--patience <n>`
- `--min-corpus-size <n>`
- `--buffer-size <n>`: online learner auto-train trigger, default `50`
- `--mini-epochs <n>`: online batch epochs, must be `1..=3`
- `--port <n>`: HTTP port in online mode
- `--pooled` or `--chunked`: embedding response mode in online mode

### `eval`

```bash
turbocalm-train eval --corpus eval-corpus.json --checkpoint latest.safetensors --device cpu
```

### `calibrate`

```bash
turbocalm-train calibrate --checkpoint latest.safetensors --corpus calibration-corpus.json
```

Outputs a `calibration.toml` file containing:

- Recommended `dedup`
- Recommended `cluster`
- Recommended `convergence`
- Per-tier similarity distribution stats for `exact`, `near-dup`, `related`, and `unrelated`

### `serve`

```bash
turbocalm-train serve --checkpoint latest.safetensors --port 11435 --pooled --device cpu
```

Available endpoints:

- `POST /v1/embeddings`
- `POST /v1/online-learn` when running `train --online`

Example request:

```json
{
  "input": "steady breathing practice",
  "model": "turbocalm-local"
}
```

### `corpus build`

```bash
turbocalm-train corpus build \
  --sources existing.jsonl,extra.jsonl \
  --memory-db ~/.openclaw/memory/lancedb-pro/memories.lance \
  --sessions ~/.openclaw/agents/main/sessions \
  --git-repos ~/Desktop/GitHub/candle-turbocalm,~/Desktop/GitHub/Triumvirate \
  --output corpus.jsonl
```

Supported inputs:

- `--sources`: existing JSONL corpora using the `CorpusEntry` format
- `--memory-db`: OpenClaw LanceDB memory path or exported backup JSONL
- `--sessions`: directory of OpenClaw `.jsonl` session logs
- `--git-repos`: one or more repos whose commit messages should be ingested

The builder deduplicates normalized text across sources before writing the unified corpus.

### `checkpoints list`

```bash
turbocalm-train checkpoints list --dir ./artifacts/checkpoints
```

## Data Formats

### Corpus JSONL

Each line is a `CorpusEntry`:

```json
{"text":"steady breathing inhale exhale","category":"breathing","timestamp":1743250000,"source":"memory:global"}
```

### Calibration Corpus

Explicit labeled pairs:

```json
{
  "pairs": [
    {"left":"a","right":"a","tier":"exact"},
    {"left":"a","right":"a paraphrase","tier":"near-dup"},
    {"left":"distributed systems","right":"event sourcing","tier":"related"},
    {"left":"distributed systems","right":"garden compost","tier":"unrelated"}
  ]
}
```

`calibrate` also accepts the existing eval-corpus shape and derives synthetic tiers from it when needed.

## Configuration Reference

### Training

- `batch_size`: contrastive batch size
- `lr`: AdamW learning rate
- `weight_decay`: AdamW weight decay
- `temperature`: NT-Xent temperature
- `max_epochs`: offline epoch limit
- `eval_interval`: checkpointing/evaluation cadence
- `patience`: early-stop patience on non-improving loss
- `checkpoint_dir`: versioned checkpoint output directory
- `min_corpus_size`: minimum pair count required to start training

### Online Learning

- `buffer_size`: number of buffered texts required before auto-training
- `mini_epochs`: online batch training epochs, capped at `3`
- `checkpoint`: optional starting checkpoint
- `port`: HTTP server port

### Serving

- `--pooled`: one vector per input text
- `--chunked`: per-chunk vectors per input text
- `model`: request must use `turbocalm-local`

## Repo Layout

```text
turbocalm-core          shared config, device, tokenizer, metrics
turbocalm-checkpoint    HF download, conversion, remapping
turbocalm-models        CALM autoencoder + language model
turbocalm-kv            dense and compressed KV cache implementations
turbocalm-calibrate     quantization calibration tooling
turbocalm-triumvirate   Triumvirate embedding and scoring adapters
turbocalm-train         corpus, training, eval, calibration, serving
turbocalm-cli           general inference CLI
```

## Notes

- Use `--device cpu` for `turbocalm-train` in this environment. Metal is intentionally disabled in the training runtime here.
- Checkpoints are written as `.safetensors` plus adjacent config JSON files.
- Online learner batches save a fresh checkpoint after every auto-training cycle.

## References

- [CALM: Continuous Autoregressive Language Models](https://arxiv.org/abs/2510.27688)
- [Candle](https://github.com/huggingface/candle)
- [PolarQuant](https://github.com/ericshwu/PolarQuant)
- [QJL](https://github.com/amirzandieh/QJL)

## License

MIT
