# Phase 3: Calibration & Validation Tasks

Complete these tasks IN ORDER. Run `cargo test` after each and fix failures.

## Task 15 (C06): Replace synthetic fitness with real model evaluation

In `crates/turbocalm-calibrate/src/objective.rs`:
- The `simulate_quantization` method uses synthetic heuristics instead of real model inference
- Replace the synthetic quality degradation with actual quantization evaluation:
  1. Take the ProcessedDataset KV traces 
  2. Quantize them with the given QuantProfile using real PolarQuantizer + QJL from turbocalm-kv
  3. Dequantize and compare with originals using cosine similarity and MSE from turbocalm-core metrics
  4. Memory gain should be computed from actual packed sizes vs original sizes
  5. Keep latency_penalty as an estimate (real latency requires actual model execution)
- Add turbocalm-kv as dependency of turbocalm-calibrate if not already present
- The reference metrics should include baseline tensor statistics from the unquantized dataset

## Task 16 (H10): Wire PolarQuantizer to respect scale_mode

In `crates/turbocalm-kv/src/quant/polar.rs`:
- PolarQuantizer always does per-token max-abs scaling
- The QuantProfile has a `scale_mode` field (or similar)
- Support at least: "per_token" (current behavior) and "per_channel" (scale across channel dim)
- If the unified QuantProfile doesn't have scale_mode, add it to turbocalm-core/src/quant.rs

## Task 17 (H11): Respect tie_word_embeddings config

In `crates/turbocalm-models/src/calm/autoencoder.rs`:
- The autoencoder always ties word embeddings (encoder/decoder share same embedding weight)
- Check the config's `tie_word_embeddings` flag
- If false, create separate embeddings for encoder and decoder lm_head

## Task 18 (M06): Fix safetensors export format

In `crates/turbocalm-calibrate/src/profiles.rs`:
- `save_as_safetensors` writes JSON data to a .safetensors file
- Either: use actual safetensors format with proper tensor serialization, OR rename to .json
- Simplest fix: rename the output file extension to .json and update the function name/docs

## Task 19 (M01): Reduce AE decoder passes in sampling

In `crates/turbocalm-models/src/calm/generation.rs`:
- `temperature_sample` (or similar) decodes num_samples (potentially 200) candidates through full AE decoder per patch
- Reduce by: scoring candidates in latent space first, then only decoding the selected winner
- Or: reduce default num_samples to something reasonable (8-16)
- Or: implement a lightweight scoring head for candidate selection

## Task 20: Add end-to-end smoke test

Create `tests/smoke_test.rs` at workspace root (or in turbocalm-cli):
- Test the full pipeline: config parsing → device selection → tokenizer loading → model init → encode → decode
- Use VarBuilder::zeros for weights (no real checkpoint download)
- But verify the complete pipeline doesn't panic and produces correct shapes
- Test that TurboKvCache produces smaller memory footprint than DenseKvCache

## Task 21: Add CMA-ES edge case tests

In `crates/turbocalm-calibrate/src/cmaes.rs`:
- Add test for mu=1 (minimum parent size) — verify no NaN
- Add test for convergence detection
- Add test that ask→tell cycle works for 100 iterations without NaN/Inf
- Add test for edge case: very large/small parameter values

## Final Step

Run `cargo test` — all must pass. Then:
```
git add -A && git commit -m "feat: Phase 3 — real calibration fitness, validation tests, smoke tests"
```
