# Phase 4: Polish Tasks

Complete these tasks. Run `cargo test` after each.

## Task 22 (M02): Standardize error handling

The workspace uses inconsistent error types:
- turbocalm-core: Custom TurboCALMError enum
- turbocalm-checkpoint: anyhow::Result
- turbocalm-calibrate: anyhow::Result  
- turbocalm-kv: candle_core::Result

**FIX:** Standardize on `anyhow::Result` across all crates for simplicity. Keep `TurboCALMError` in core for type-specific matching when needed but use anyhow for public interfaces. Add `.context()` calls for better error messages at crate boundaries. Remove any redundant error conversions.

## Task 23 (M03): Fix memory reporting

In `crates/turbocalm-core/src/metrics.rs`:
- `get_macos_available_memory()` returns hardcoded 1024.0 MB
- **FIX:** Actually parse `vm_stat` output to compute free+inactive pages, multiply by page size (usually 16384 on ARM64)
- Also improve `get_macos_rss_mb()` to be more robust

## Task 24 (M04): Relax temperature validation

In `crates/turbocalm-models/src/calm/generation.rs`:
- Temperature is forced to be reciprocal of integer (validate_temperature rejects T=0.7)
- **FIX:** Allow any positive temperature value. Just validate it's > 0 and not NaN/Inf
- Update the generation config to use f64 temperature directly

## Task 25 (M05): Implement Triumvirate adapter or remove crate

In `crates/turbocalm-triumvirate/`:
- Currently empty (single comment line)
- **FIX:** Add basic adapter structs with trait definitions for:
  - `EmbeddingEngine` — wraps CalmAutoencoder for Triumvirate's embedding interface
  - `EnergyScorer` — wraps CalmLanguageModel for Triumvirate's scoring interface
- Each should have a `new()` constructor taking a config path/model ID
- These can be stubs with clear interfaces, but not completely empty

## Task 26 (H04): Document PyTorch .bin limitation

In `crates/turbocalm-checkpoint/src/parser.rs`:
- PyTorch .bin loading returns error with suggestion to convert
- **FIX:** Improve the error message to include the specific conversion command
- Add a section to the README about supported checkpoint formats
- In the CLI help text for convert command, mention .bin → safetensors conversion

## Task 27 (H01): Wire remaining CLI commands

In `crates/turbocalm-cli/src/main.rs`:
- Score, Generate, Calibrate, Benchmark still use todo!()
- **FIX:** Replace todo!() with proper error messages like:
  "Score command requires a loaded model. Use: turbocalm score --model <HF_MODEL_ID> --text <TEXT>"
  Then print what would need to happen (download, load, tokenize, score)
  Add --dry-run flag concept
- Don't implement the full functionality — just make the commands NOT PANIC
- Print helpful "not yet implemented" messages with prerequisites

## Final Step

Run `cargo test` — all must pass. Clean up any compiler warnings with `cargo fix --allow-dirty`. Then:
```
git add -A && git commit -m "feat: Phase 4 — polish (error handling, memory reporting, temperature, Triumvirate adapter, CLI)"
```

After committing, also:
1. Delete PHASE2_TASKS.md, PHASE3_TASKS.md, PHASE4_TASKS.md
2. Update README.md to reflect current state
3. Final commit: git add -A && git commit -m "chore: cleanup task files, update README"
