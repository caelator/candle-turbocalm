# KV Fusion Spike: Latent Space Communication

## Objective
Implement a mathematical fusion operation in `candle-turbocalm` that takes multiple `CompressedTensor` KV caches (representing the hidden state "thoughts" of different Acolyte agents) and merges them into a single coherent KV cache. 

This enables agents to communicate their raw understanding (vectors) rather than forcing them to serialize their thoughts into text tokens.

## Architecture

We need to add a `fuse` operation to `TurboKvCache` in `crates/turbocalm-kv/src/cache/mod.rs`.

### The Problem
If Agent A produces KV cache $K_A, V_A$ (shape: `[batch, heads, seq_len_A, head_dim]`) and Agent B produces $K_B, V_B$ (shape: `[batch, heads, seq_len_B, head_dim]`), we cannot simply concatenate them if we want a fixed-size compressed "consensus" state. 

### The Solution: Attention Pooling (The Spike)
We will implement a simple attention-pooling mechanism to fuse multiple KV caches down to a target sequence length (e.g., merging 3 agents' thoughts into a single "consensus" KV cache).

1. **Decompress:** Unpack the `CompressedTensor` objects back to `f32` or `f16`.
2. **Concatenate:** Join the key and value tensors along the sequence dimension.
3. **Pool:** Apply a pooling operation (e.g., mean pooling or learned attention pooling) over the sequence dimension to compress the fused thoughts down to a manageable size.
4. **Re-compress:** Pass the fused, pooled tensor back through `TurboQuant` to generate the final `CompressedTensor`.

## Codex Task
1. Modify `crates/turbocalm-kv/src/cache/mod.rs` to add a `fuse_caches(caches: &[&TurboKvCache], target_seq_len: usize) -> Result<TurboKvCache>` function.
2. Implement the unpacking, concatenation, mean-pooling (down to `target_seq_len`), and re-packing logic.
3. Write a test in `crates/turbocalm-kv/tests/` that:
   - Generates two synthetic KV caches.
   - Compresses them.
   - Fuses them using the new function.
   - Verifies the output shape matches the target and the values are mathematically sound.
