# Candle-TurboCALM

Native Rust implementation of [CALM (Continuous Autoregressive Language Models)](https://arxiv.org/abs/2510.27688) powered by [Candle](https://github.com/huggingface/candle) with Metal GPU acceleration and TurboQuant KV cache compression.

## Features

- 🦀 **Pure Rust** — no Python runtime dependency
- 🍎 **Apple Silicon native** — Metal GPU acceleration via Candle
- 🗜️ **TurboQuant compression** — PolarQuant + QJL for ~6x KV cache reduction
- 🧬 **Evolutionary calibration** — gradient-free optimization of quantization parameters
- 📦 **Pretrained checkpoints** — loads CALM-Autoencoder, CALM-M, CALM-L from HuggingFace

## Quick Start

```bash
# Build with Metal support
cargo build --release

# Encode text to CALM embeddings
turbocalm encode --model cccczshao/CALM-Autoencoder --text "Hello world"

# Score text with energy model
turbocalm score --model cccczshao/CALM-M --text "The quick brown fox"

# Generate text
turbocalm generate --model cccczshao/CALM-M --prompt "Once upon a time" --max-tokens 256

# Run evolutionary calibration
turbocalm calibrate --model cccczshao/CALM-M --corpus ./calibration/corpus.jsonl --profile balanced
```

## Architecture

```
turbocalm-core          — shared config, device, tokenizer, metrics
turbocalm-checkpoint    — HF download, .bin→safetensors conversion, weight remapping
turbocalm-models        — CALM autoencoder (MLP) + language model (attention + energy)
turbocalm-kv            — DenseKvCache + TurboKvCache (PolarQuant + QJL)
turbocalm-calibrate     — CMA-ES evolutionary engine for quantization parameter tuning
turbocalm-triumvirate   — EmbeddingEngine + EnergyScorer adapters for Triumvirate integration
turbocalm-cli           — CLI: inspect, convert, encode, score, generate, calibrate, benchmark
```

## Pretrained Models

| Model | Parameters | Purpose |
|-------|:---:|---------|
| [CALM-Autoencoder](https://huggingface.co/cccczshao/CALM-Autoencoder) | 75M | Embedding (encode text → 128-dim vectors) |
| [CALM-M](https://huggingface.co/cccczshao/CALM-M) | 371M | Energy scoring + generation |
| [CALM-L](https://huggingface.co/cccczshao/CALM-L) | 735M | Higher quality scoring + generation |
| [CALM-XL](https://huggingface.co/cccczshao/CALM-XL) | 1.82B | Research-quality evaluation |

## References

- [CALM: Continuous Autoregressive Language Models](https://arxiv.org/abs/2510.27688) (Shao et al., 2025)
- [Candle](https://github.com/huggingface/candle) (Hugging Face)
- [PolarQuant](https://github.com/ericshwu/PolarQuant) (sub-byte KV cache quantization)
- [QJL](https://github.com/amirzandieh/QJL) (1-bit residual projection)

## License

MIT
