# CALM Metal Training Spike

- Auto-selected device: `metal`
- Metal available: `true`
- Fell back to CPU: `no`

## Metal Attempt

- Requested device: `metal`
- Actual device used: `metal`
- Success: `true`
- Loss: `0.009862`
- Forward time: `4044 ms`
- Backward time: `112 ms`
- AdamW step time: `1 ms`
- Watched var: `encoder.norm.weight`
- Max abs weight delta: `0.01000000`
- Changed var count: `2`
- Weights changed after step: `true`

## Notes

- Trainable weights come from `VarMap` + `VarBuilder::from_varmap`, not `VarBuilder::zeros`.
- The existing autoencoder loader checks `contains_tensor` for embedding paths, so the spike pre-seeds `encoder.embed_tokens.weight` before calling `CalmAutoencoder::load`.
- The RMSNorm path exercised here is the manual pure-tensor implementation in `autoencoder.rs`, not Candle's custom RMSNorm op.
