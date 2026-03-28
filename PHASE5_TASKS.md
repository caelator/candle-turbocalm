# Phase 5: Functional Completion

Complete these tasks IN ORDER. Run `cargo test` after each.

## Task 1: Wire CLI inspect command

In `crates/turbocalm-cli/src/main.rs`, the Inspect command prints TODO.

**FIX:**
- Use `turbocalm_core::HubClient` to download `config.json` from the HF model
- Parse it as `CALMConfig` or generic JSON
- Print a summary: model_type, hidden_size, num_layers, num_attention_heads, vocab_size, patch_size, latent_size
- If the model also has an autoencoder config, print that too
- Handle errors gracefully (network failures, invalid model IDs)

## Task 2: Wire CLI convert command

The Convert command prints TODO.

**FIX:**
- Use `turbocalm_checkpoint::ConvertHandler` or build the conversion pipeline:
  1. Download model files from HF using HubClient
  2. Parse checkpoint files using StateDictParser
  3. Remap tensor names using RemappingUtils
  4. Save as safetensors to the output path
- If the model is already safetensors, just copy/verify
- Print progress messages during conversion

## Task 3: Wire CLI encode command

The Encode command prints TODO.

**FIX:**
- Load tokenizer from HF model using TokenizerLoader
- Tokenize the input text
- Create a CalmAutoencoder with VarBuilder::zeros (since we don't have real weights without download)
- Actually, for the encode command to work properly with real weights:
  1. Download model checkpoint
  2. Load autoencoder weights  
  3. Encode text → latent embedding
  4. Print embedding shape and first few values
- If weights not available, print the token IDs and expected embedding shape

## Task 4: Wire CLI score command

Replace the "not yet implemented" message in Score.

**FIX:**
- Build the full scoring pipeline:
  1. Download model + tokenizer from HF
  2. Load CalmLanguageModel with real weights
  3. Tokenize input text
  4. Encode through autoencoder
  5. Score with language model energy function
  6. Print the energy score
- If this is too complex for real weights, at minimum:
  - Download and load config
  - Initialize model with zeroed weights
  - Run the full pipeline (won't give meaningful scores but proves the pipeline works)
  - Print shapes and dummy scores with a note

## Task 5: Wire CLI generate command

Replace the "not yet implemented" message in Generate.

**FIX:**
- Build the full generation pipeline:
  1. Download model + tokenizer from HF  
  2. Load CalmGenerationModel
  3. Tokenize prompt
  4. Run generate() with provided max_tokens
  5. Decode and print generated text
- Same approach as score: if real weights are impractical, demonstrate the pipeline with zeroed weights

## Task 6: Implement Triumvirate adapter methods

In `crates/turbocalm-triumvirate/src/lib.rs`, all methods bail with "not yet implemented".

**FIX:**
For EmbeddingEngine:
- `load()` should download checkpoint, load autoencoder config, create CalmAutoencoder
- `embed()` should tokenize text, run autoencoder encode, return latent tensor
- `embedding_dim()` should return the config's latent_size

For EnergyScorer:
- `load()` should download checkpoint, load LM config, create CalmLanguageModel
- `score()` should tokenize, encode, run LM forward, return energy score
- `latent_dim()` should return config's latent_size

Use turbocalm_core::HubClient and turbocalm_models for actual implementations.
Add turbocalm-models and turbocalm-checkpoint as dependencies.

## Task 7: Wire CLI calibrate command

Replace "not yet implemented" in Calibrate.

**FIX:**
- Build the calibration pipeline:
  1. Load calibration dataset from JSONL corpus file
  2. Load model config from HF
  3. Create CalibrationSearch with the specified profile (rapid/balanced/thorough)
  4. Run search
  5. Export results using ProfileExporter
  6. Print summary report
- This should work even without real model weights since calibration evaluates quantization quality

## Task 8: Wire CLI benchmark command

Replace "not yet implemented" in Benchmark.

**FIX:**
- Build a basic benchmark:
  1. Load model config from HF
  2. Create model with VarBuilder::zeros
  3. Run encode/decode/generate with sample inputs
  4. Time each operation
  5. Compare DenseKvCache vs TurboKvCache memory usage
  6. Print results in a table format

## Task 9: Clean up remaining Phase 5 TODOs in calibrate

In `crates/turbocalm-calibrate/src/objective.rs`, there are 5 `TODO(Phase 5)` comments for:
- Real output quality metrics (Brier score)
- Real activation tensor comparison
- Real baseline metrics
- Real latency measurement

**FIX:**
- For quality: use the real quantize→dequantize cosine similarity already wired in Phase 3
- For Brier score: implement a simple Brier-like score from softmax output probabilities
- For latency: time the actual quantize/dequantize operations
- Replace synthetic estimates with real measurements where possible
- Mark any truly infeasible items (e.g., full model inference latency) with clear "requires full model" annotations

## Final Step

Run `cargo test` — all must pass. Then:
```
git add -A && git commit -m "feat: Phase 5 — functional completion (CLI wiring, Triumvirate adapter, calibration pipeline)"
rm PHASE5_TASKS.md && git add -A && git commit -m "chore: cleanup task file"
```
