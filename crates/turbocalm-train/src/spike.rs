use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::{AdamW, Init, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use turbocalm_models::{CalmAutoencoder, CalmAutoencoderConfig};

const RESULTS_PATH: &str = "SPIKE-RESULTS.md";
const PREFERRED_WATCH_VAR_NAME: &str = "encoder.hidden_to_latent.weight";

#[derive(Debug, Clone)]
pub struct DeviceAttempt {
    pub requested: &'static str,
    pub actual: String,
    pub success: bool,
    pub weight_changed: bool,
    pub forward_ms: u128,
    pub backward_ms: u128,
    pub step_ms: u128,
    pub loss_value: Option<f32>,
    pub watched_var: Option<String>,
    pub max_abs_weight_delta: Option<f32>,
    pub changed_var_count: usize,
    pub failure_stage: Option<&'static str>,
    pub failing_backward_op: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SpikeReport {
    pub metal_attempted: bool,
    pub metal_available: bool,
    pub metal: Option<DeviceAttempt>,
    pub cpu: Option<DeviceAttempt>,
    pub fell_back_to_cpu: bool,
    pub auto_device: String,
    pub notes: Vec<String>,
}

impl SpikeReport {
    pub fn render_console(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "auto-device: {}", self.auto_device);
        let _ = writeln!(out, "metal-available: {}", self.metal_available);
        let _ = writeln!(
            out,
            "fell-back-to-cpu: {}",
            if self.fell_back_to_cpu { "yes" } else { "no" }
        );
        if let Some(metal) = &self.metal {
            let _ = writeln!(out, "metal: {}", summarize_attempt(metal));
        }
        if let Some(cpu) = &self.cpu {
            let _ = writeln!(out, "cpu: {}", summarize_attempt(cpu));
        }
        for note in &self.notes {
            let _ = writeln!(out, "note: {note}");
        }
        out
    }

    pub fn render_markdown(&self) -> String {
        let mut out = String::new();
        out.push_str("# CALM Metal Training Spike\n\n");
        let _ = writeln!(out, "- Auto-selected device: `{}`", self.auto_device);
        let _ = writeln!(out, "- Metal available: `{}`", self.metal_available);
        let _ = writeln!(
            out,
            "- Fell back to CPU: `{}`",
            if self.fell_back_to_cpu { "yes" } else { "no" }
        );

        if let Some(metal) = &self.metal {
            out.push_str("\n## Metal Attempt\n\n");
            out.push_str(&attempt_markdown(metal));
        }

        if let Some(cpu) = &self.cpu {
            out.push_str("\n## CPU Attempt\n\n");
            out.push_str(&attempt_markdown(cpu));
        }

        if !self.notes.is_empty() {
            out.push_str("\n## Notes\n\n");
            for note in &self.notes {
                let _ = writeln!(out, "- {note}");
            }
        }

        out
    }
}

pub fn run_and_write_results() -> Result<SpikeReport> {
    let report = run()?;
    std::fs::write(PathBuf::from(RESULTS_PATH), report.render_markdown())
        .with_context(|| format!("failed to write {RESULTS_PATH}"))?;
    Ok(report)
}

pub fn run() -> Result<SpikeReport> {
    let auto_device = match catch_panic(|| turbocalm_core::device::auto_device()) {
        Ok(Ok(device)) => device_name(&device),
        Ok(Err(error)) => format!("auto-device-error: {error}"),
        Err(panic) => format!("auto-device-panic: {}", panic.render()),
    };

    let mut notes = vec![
        "Trainable weights come from `VarMap` + `VarBuilder::from_varmap`, not `VarBuilder::zeros`.".to_string(),
        "The existing autoencoder loader checks `contains_tensor` for embedding paths, so the spike pre-seeds `encoder.embed_tokens.weight` before calling `CalmAutoencoder::load`.".to_string(),
        "The RMSNorm path exercised here is the manual pure-tensor implementation in `autoencoder.rs`, not Candle's custom RMSNorm op.".to_string(),
    ];

    let metal_device = match catch_panic(|| Device::new_metal(0)) {
        Ok(result) => result.map_err(|error| error.to_string()),
        Err(panic) => Err(panic.render()),
    };
    let metal_available = metal_device.is_ok();
    let metal = match metal_device {
        Ok(device) => Some(run_attempt_guarded("metal", device)?),
        Err(error) => {
            notes.push(format!("Metal device creation failed: {error}"));
            Some(device_failure_attempt("metal", "device-create", error))
        }
    };

    let need_cpu_fallback = match metal.as_ref() {
        Some(attempt) => !attempt.success,
        None => true,
    };

    let cpu = if need_cpu_fallback {
        Some(run_attempt("cpu", Device::Cpu)?)
    } else {
        None
    };

    Ok(SpikeReport {
        metal_attempted: true,
        metal_available,
        metal,
        cpu,
        fell_back_to_cpu: need_cpu_fallback,
        auto_device,
        notes,
    })
}

fn run_attempt_guarded(requested: &'static str, device: Device) -> Result<DeviceAttempt> {
    match catch_panic(move || run_attempt(requested, device)) {
        Ok(result) => result,
        Err(panic) => Ok(device_failure_attempt(requested, "panic", panic.render())),
    }
}

fn run_attempt(requested: &'static str, device: Device) -> Result<DeviceAttempt> {
    let actual = device_name(&device);
    let config = spike_config();
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    preseed_embeddings(&vb, &config)
        .with_context(|| format!("failed to pre-seed trainable embeddings on {requested}"))?;

    let model = CalmAutoencoder::load(vb.clone(), config.clone())
        .with_context(|| format!("failed to build CalmAutoencoder on {requested}"))?;

    let watched_var = choose_watched_var(&varmap)
        .context("failed to locate a trainable variable to compare before/after optimizer step")?;
    let before = snapshot_all_vars(&varmap)?;

    let input = dummy_input(&device)?;

    let started = Instant::now();
    let embedding = match model.encode_pooled(&input) {
        Ok(embedding) => embedding,
        Err(error) => {
            return Ok(DeviceAttempt {
                requested,
                actual,
                success: false,
                weight_changed: false,
                forward_ms: started.elapsed().as_millis(),
                backward_ms: 0,
                step_ms: 0,
                loss_value: None,
                watched_var: Some(watched_var.0),
                max_abs_weight_delta: None,
                changed_var_count: 0,
                failure_stage: Some("forward"),
                failing_backward_op: None,
                error: Some(format!("{error:#}")),
            });
        }
    };
    let forward_ms = started.elapsed().as_millis();

    let loss = embedding
        .sqr()
        .context("failed to square pooled embedding")?
        .mean_all()
        .context("failed to reduce pooled embedding loss")?;
    let loss_value = scalar_f32(&loss)?;

    let backward_started = Instant::now();
    let grads = match loss.backward() {
        Ok(grads) => grads,
        Err(error) => {
            let error_text = format!("{error:#}");
            return Ok(DeviceAttempt {
                requested,
                actual,
                success: false,
                weight_changed: false,
                forward_ms,
                backward_ms: backward_started.elapsed().as_millis(),
                step_ms: 0,
                loss_value: Some(loss_value),
                watched_var: Some(watched_var.0),
                max_abs_weight_delta: None,
                changed_var_count: 0,
                failure_stage: Some("backward"),
                failing_backward_op: infer_failing_op(&error_text),
                error: Some(error_text),
            });
        }
    };
    let backward_ms = backward_started.elapsed().as_millis();

    let mut optimizer = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: 1e-2,
            weight_decay: 0.0,
            ..ParamsAdamW::default()
        },
    )
    .context("failed to create AdamW optimizer")?;

    let step_started = Instant::now();
    if let Err(error) = optimizer.step(&grads) {
        let error_text = format!("{error:#}");
        return Ok(DeviceAttempt {
            requested,
            actual,
            success: false,
            weight_changed: false,
            forward_ms,
            backward_ms,
            step_ms: step_started.elapsed().as_millis(),
            loss_value: Some(loss_value),
            watched_var: Some(watched_var.0),
            max_abs_weight_delta: None,
            changed_var_count: 0,
            failure_stage: Some("optimizer-step"),
            failing_backward_op: infer_failing_op(&error_text),
            error: Some(error_text),
        });
    }
    let step_ms = step_started.elapsed().as_millis();

    let diff_summary = summarize_var_deltas(&varmap, &before)?;
    let weight_changed = diff_summary.changed_var_count > 0;
    let failure_stage = if weight_changed {
        None
    } else {
        Some("verification")
    };
    let error = if weight_changed {
        None
    } else {
        Some("AdamW step completed but no tracked trainable variable changed".to_string())
    };

    Ok(DeviceAttempt {
        requested,
        actual,
        success: weight_changed,
        weight_changed,
        forward_ms,
        backward_ms,
        step_ms,
        loss_value: Some(loss_value),
        watched_var: Some(diff_summary.top_var.unwrap_or(watched_var.0)),
        max_abs_weight_delta: Some(diff_summary.top_delta),
        changed_var_count: diff_summary.changed_var_count,
        failure_stage,
        failing_backward_op: None,
        error,
    })
}

fn device_failure_attempt(
    requested: &'static str,
    stage: &'static str,
    error: String,
) -> DeviceAttempt {
    DeviceAttempt {
        requested,
        actual: "unavailable".to_string(),
        success: false,
        weight_changed: false,
        forward_ms: 0,
        backward_ms: 0,
        step_ms: 0,
        loss_value: None,
        watched_var: None,
        max_abs_weight_delta: None,
        changed_var_count: 0,
        failure_stage: Some(stage),
        failing_backward_op: infer_failing_op(&error),
        error: Some(error),
    }
}

fn spike_config() -> CalmAutoencoderConfig {
    CalmAutoencoderConfig {
        vocab_size: 64,
        hidden_size: 32,
        intermediate_size: 64,
        latent_size: 16,
        patch_size: 4,
        num_encoder_layers: 2,
        num_decoder_layers: 2,
        tie_word_embeddings: true,
        ..Default::default()
    }
}

fn preseed_embeddings(vb: &VarBuilder<'_>, config: &CalmAutoencoderConfig) -> Result<()> {
    let init = Init::Randn {
        mean: 0.0,
        stdev: config.initializer_range,
    };

    vb.pp("encoder")
        .get_with_hints(
            (config.vocab_size, config.hidden_size),
            "embed_tokens.weight",
            init,
        )
        .context("failed to initialize encoder.embed_tokens.weight")?;

    Ok(())
}

fn dummy_input(device: &Device) -> Result<Tensor> {
    Tensor::from_vec(
        vec![
            1u32, 2, 3, 4, 5, 6, 7, 8, //
            8, 7, 6, 5, 4, 3, 2, 1,
        ],
        (2, 8),
        device,
    )
    .context("failed to create dummy token input")
}

fn choose_watched_var(varmap: &VarMap) -> Result<(String, Var)> {
    let vars = ordered_vars(varmap);
    vars.into_iter()
        .find(|(name, _)| name == PREFERRED_WATCH_VAR_NAME)
        .or_else(|| ordered_vars(varmap).into_iter().next())
        .context("varmap is empty")
}

fn ordered_vars(varmap: &VarMap) -> Vec<(String, Var)> {
    let data = varmap.data().lock().unwrap();
    let mut ordered: BTreeMap<String, Var> = BTreeMap::new();
    for (name, var) in data.iter() {
        ordered.insert(name.clone(), var.clone());
    }
    ordered.into_iter().collect()
}

fn snapshot_all_vars(varmap: &VarMap) -> Result<BTreeMap<String, Vec<f32>>> {
    let mut snapshots = BTreeMap::new();
    for (name, var) in ordered_vars(varmap) {
        snapshots.insert(name, snapshot_var(&var)?);
    }
    Ok(snapshots)
}

fn snapshot_var(var: &Var) -> Result<Vec<f32>> {
    let cpu_tensor = var
        .as_tensor()
        .to_device(&Device::Cpu)
        .context("failed to copy watched var to CPU")?;
    cpu_tensor
        .flatten_all()
        .context("failed to flatten watched var")?
        .to_vec1::<f32>()
        .context("failed to materialize watched var values")
}

fn scalar_f32(tensor: &Tensor) -> Result<f32> {
    tensor
        .to_device(&Device::Cpu)
        .context("failed to copy scalar tensor to CPU")?
        .to_scalar::<f32>()
        .context("failed to read scalar tensor")
}

#[derive(Debug)]
struct VarDiffSummary {
    top_var: Option<String>,
    top_delta: f32,
    changed_var_count: usize,
}

fn summarize_var_deltas(
    varmap: &VarMap,
    before: &BTreeMap<String, Vec<f32>>,
) -> Result<VarDiffSummary> {
    let mut top_var = None;
    let mut top_delta = 0.0f32;
    let mut changed_var_count = 0usize;

    for (name, var) in ordered_vars(varmap) {
        let after = snapshot_var(&var)?;
        if let Some(before_values) = before.get(&name) {
            let delta = max_abs_delta(before_values, &after);
            if delta > 0.0 {
                changed_var_count += 1;
            }
            if delta > top_delta {
                top_delta = delta;
                top_var = Some(name);
            }
        }
    }

    Ok(VarDiffSummary {
        top_var,
        top_delta,
        changed_var_count,
    })
}

fn max_abs_delta(before: &[f32], after: &[f32]) -> f32 {
    before
        .iter()
        .zip(after.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0f32, f32::max)
}

fn infer_failing_op(error: &str) -> Option<String> {
    [
        "index-select",
        "index_add",
        "index-add",
        "scatter-add",
        "matmul",
        "sqrt",
        "silu",
        "broadcast",
        "mean",
        "sum",
        "mul",
        "div",
    ]
    .iter()
    .find(|needle| error.contains(**needle))
    .map(|needle| needle.to_string())
}

fn device_name(device: &Device) -> String {
    if device.is_metal() {
        "metal".to_string()
    } else if device.is_cpu() {
        "cpu".to_string()
    } else if device.is_cuda() {
        "cuda".to_string()
    } else {
        format!("{device:?}")
    }
}

#[derive(Debug, Clone)]
struct PanicSummary {
    message: String,
    location: Option<String>,
}

impl PanicSummary {
    fn render(&self) -> String {
        match &self.location {
            Some(location) => format!("{} at {}", self.message, location),
            None => self.message.clone(),
        }
    }
}

fn catch_panic<F, T>(f: F) -> std::result::Result<T, PanicSummary>
where
    F: FnOnce() -> T,
{
    let captured = Arc::new(Mutex::new(None::<PanicSummary>));
    let hook_capture = captured.clone();
    let previous_hook = std::panic::take_hook();

    std::panic::set_hook(Box::new(move |info| {
        let message = if let Some(message) = info.payload().downcast_ref::<&str>() {
            (*message).to_string()
        } else if let Some(message) = info.payload().downcast_ref::<String>() {
            message.clone()
        } else {
            "panic payload was not a string".to_string()
        };
        let location = info
            .location()
            .map(|location| format!("{}:{}", location.file(), location.line()));
        *hook_capture.lock().unwrap() = Some(PanicSummary { message, location });
    }));

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
    std::panic::set_hook(previous_hook);

    match result {
        Ok(value) => Ok(value),
        Err(_) => Err(captured.lock().unwrap().clone().unwrap_or(PanicSummary {
            message: "panic without captured message".to_string(),
            location: None,
        })),
    }
}

fn summarize_attempt(attempt: &DeviceAttempt) -> String {
    if attempt.success {
        format!(
            "ok actual={} loss={:.6} changed={} delta={:.6} forward={}ms backward={}ms step={}ms changed_vars={} var={}",
            attempt.actual,
            attempt.loss_value.unwrap_or_default(),
            attempt.weight_changed,
            attempt.max_abs_weight_delta.unwrap_or_default(),
            attempt.forward_ms,
            attempt.backward_ms,
            attempt.step_ms,
            attempt.changed_var_count,
            attempt
                .watched_var
                .as_deref()
                .unwrap_or("<unknown>")
        )
    } else {
        format!(
            "failed actual={} stage={} op={} error={}",
            attempt.actual,
            attempt.failure_stage.unwrap_or("unknown"),
            attempt.failing_backward_op.as_deref().unwrap_or("unknown"),
            attempt.error.as_deref().unwrap_or("<missing>")
        )
    }
}

fn attempt_markdown(attempt: &DeviceAttempt) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "- Requested device: `{}`", attempt.requested);
    let _ = writeln!(out, "- Actual device used: `{}`", attempt.actual);
    let _ = writeln!(out, "- Success: `{}`", attempt.success);
    if let Some(loss_value) = attempt.loss_value {
        let _ = writeln!(out, "- Loss: `{loss_value:.6}`");
    }
    let _ = writeln!(out, "- Forward time: `{} ms`", attempt.forward_ms);
    let _ = writeln!(out, "- Backward time: `{} ms`", attempt.backward_ms);
    let _ = writeln!(out, "- AdamW step time: `{} ms`", attempt.step_ms);
    if let Some(watched_var) = &attempt.watched_var {
        let _ = writeln!(out, "- Watched var: `{watched_var}`");
    }
    if let Some(delta) = attempt.max_abs_weight_delta {
        let _ = writeln!(out, "- Max abs weight delta: `{delta:.8}`");
    }
    let _ = writeln!(out, "- Changed var count: `{}`", attempt.changed_var_count);
    let _ = writeln!(
        out,
        "- Weights changed after step: `{}`",
        attempt.weight_changed
    );
    if let Some(stage) = attempt.failure_stage {
        let _ = writeln!(out, "- Failure stage: `{stage}`");
    }
    if let Some(op) = &attempt.failing_backward_op {
        let _ = writeln!(out, "- Failing backward op: `{op}`");
    }
    if let Some(error) = &attempt.error {
        out.push_str("- Error:\n\n```text\n");
        out.push_str(error);
        out.push_str("\n```\n");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_spike_updates_weights() -> Result<()> {
        let attempt = run_attempt("cpu", Device::Cpu)?;
        assert!(attempt.success, "cpu attempt failed: {:?}", attempt.error);
        assert!(
            attempt.weight_changed,
            "optimizer step did not change weights"
        );
        Ok(())
    }
}
