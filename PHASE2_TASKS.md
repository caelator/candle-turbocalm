# Phase 2: Core Wiring Tasks

Complete these tasks IN ORDER. After each task, run `cargo test` and fix any failures before moving to the next.

## Task 8 (H02): Unify LM implementations

There are TWO transformer implementations that overlap:
1. `CalmLanguageModel` in `crates/turbocalm-models/src/calm/lm.rs` — uses `self_attn` naming
2. The transformer in `crates/turbocalm-models/src/calm/generation.rs` — uses `attention` naming

**FIX:** Keep generation.rs as the primary implementation (it has the full pipeline). Simplify lm.rs to be a thin wrapper or re-export of generation.rs types. If lm.rs types are used elsewhere, update those references. Align tensor naming. Update exports in `crates/turbocalm-models/src/lib.rs`. Keep all existing tests passing.

## Task 9 (C01): Wire latent conditioning into transformer

Currently `CalmGenerationModel::generate()` computes `prompt_latents` via autoencoder but the transformer re-embeds raw tokens. The transformer should consume latent patches instead.

**FIX:** Make the generation pipeline actually feed encoded latent representations to the transformer rather than raw token embeddings. The autoencoder encodes tokens into latent patches; the transformer should process these.

## Task 10 (C04): Wire TurboKvCache into model attention

The `KvCache` trait is defined in `crates/turbocalm-kv/src/cache/mod.rs`. `TurboKvCache` implements it. But `Attention` in generation.rs uses `Option<(Tensor, Tensor)>`.

**FIX:** Add `turbocalm-kv` as dependency of `turbocalm-models`. Make attention use the `KvCache` trait or `TurboKvCache` directly for KV storage/retrieval.

## Task 11 (H09): Integrate bit-packing into TurboKvCache

`pack_bits`/`unpack_bits` in `crates/turbocalm-kv/src/quant/pack.rs` exist but are never called. `TurboKvCache` stores quantized values as full f32.

**FIX:** Apply `pack_bits()` after quantization in `compress_tensor`. Apply `unpack_bits()` before dequantization in `decompress_tensor`.

## Task 12 (C03): Fix O(N²) cache retrieval

`get_key`/`get_value` decompress ALL stored tensors on every call.

**FIX:** Cache the uncompressed prefix. Only decompress new entries since last retrieval, then concat with cached result.

## Task 13 (H03): Unify QuantProfile types

Two `QuantProfile` structs exist:
1. `crates/turbocalm-kv/src/quant/profile.rs` 
2. `crates/turbocalm-calibrate/src/lib.rs`

**FIX:** Move a unified `QuantProfile` to `turbocalm-core`. Include all fields from both. Update imports everywhere.

## Task 14 (H01): Wire CLI inspect/convert/encode

In `crates/turbocalm-cli/src/main.rs`, 3 commands print TODO.

**FIX:** 
- Inspect: Download config.json from HF, parse, print summary
- Convert: Wire to ConvertHandler  
- Encode: Load tokenizer + autoencoder, encode text, print embedding dims
- Keep Score/Generate/Calibrate/Benchmark as todo!() for now

## Final Step

After ALL tasks, run `cargo test` and ensure everything passes. Then:
```
git add -A && git commit -m "feat: Phase 2 — core wiring (LM unification, latent conditioning, KvCache integration, CLI)"
```
