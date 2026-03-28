use clap::{Parser, Subcommand};
use anyhow::Result;
use candle_core::{DType, Tensor};
use std::path::PathBuf;
use turbocalm_core::{auto_device, DownloadUtils, TokenizerLoader, TokenizerType, TokenizerUtils, AutoencoderConfig};
use turbocalm_models::{CalmAutoencoder, CalmAutoencoderConfig};
use turbocalm_checkpoint::{run_convert_command, ConvertArgs, ConvertCommand};
use turbocalm_checkpoint::convert::FromHubArgs;

#[derive(Parser)]
#[command(name = "turbocalm", about = "Candle-TurboCALM: Native CALM inference with TurboQuant compression")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Inspect a CALM checkpoint
    Inspect {
        #[arg(long)]
        model: String,
    },
    /// Convert PyTorch checkpoint to safetensors
    Convert {
        #[arg(long)]
        model: String,
        #[arg(long)]
        output: String,
    },
    /// Encode text to CALM embeddings
    Encode {
        #[arg(long)]
        model: String,
        #[arg(long)]
        text: String,
    },
    /// Score text with CALM energy model
    Score {
        #[arg(long)]
        model: String,
        #[arg(long)]
        text: String,
    },
    /// Generate text using CALM
    Generate {
        #[arg(long)]
        model: String,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value = "256")]
        max_tokens: usize,
    },
    /// Run TurboQuant evolutionary calibration
    Calibrate {
        #[arg(long)]
        model: String,
        #[arg(long)]
        corpus: String,
        #[arg(long, default_value = "balanced")]
        profile: String,
    },
    /// Benchmark inference performance
    Benchmark {
        #[arg(long)]
        model: String,
    },
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Inspect { model } => {
            inspect_model(&model)
        }
        Commands::Convert { model, output } => {
            convert_checkpoint(&model, &output)
        }
        Commands::Encode { model, text } => {
            encode_text(&model, &text)
        }
        Commands::Score { model, text } => {
            score_text(&model, &text)
        }
        Commands::Generate { model, prompt, max_tokens } => {
            println!("Generating with {model} (max {max_tokens} tokens): {prompt}");
            todo!("Phase 3: native generation")
        }
        Commands::Calibrate { model, corpus, profile } => {
            println!("Calibrating {model} with {corpus} for {profile} profile");
            todo!("Phase 5: evolutionary calibration")
        }
        Commands::Benchmark { model } => {
            println!("Benchmarking {model}");
            todo!("Phase 3: benchmarking")
        }
    }
}

fn inspect_model(model_id: &str) -> Result<()> {
    println!("Inspecting model: {}", model_id);

    // Download the complete model (safetensors + config + tokenizer)
    let download = DownloadUtils::download_complete_model(model_id)?;

    println!("✓ Downloaded model files:");
    for (i, path) in download.model_paths().iter().enumerate() {
        println!("  - Model file {}: {}", i + 1, path.display());
    }

    if let Some(config_path) = download.config_path() {
        println!("  - Config: {}", config_path.display());

        // Try to load and display basic config info
        if let Ok(config_content) = std::fs::read_to_string(config_path) {
            if let Ok(config_json) = serde_json::from_str::<serde_json::Value>(&config_content) {
                if let Some(model_type) = config_json.get("model_type") {
                    println!("  - Model type: {}", model_type);
                }
                if let Some(vocab_size) = config_json.get("vocab_size") {
                    println!("  - Vocab size: {}", vocab_size);
                }
                if let Some(hidden_size) = config_json.get("hidden_size") {
                    println!("  - Hidden size: {}", hidden_size);
                }
            }
        }
    }

    if let Some(tokenizer_path) = download.tokenizer_path() {
        println!("  - Tokenizer: {}", tokenizer_path.display());
    }

    println!("\n✓ Model inspection complete");
    Ok(())
}

fn convert_checkpoint(model_id: &str, output_path: &str) -> Result<()> {
    println!("Converting {} -> {}", model_id, output_path);

    let args = ConvertArgs {
        command: ConvertCommand::FromHub(FromHubArgs {
            model_id: model_id.to_string(),
            output: Some(PathBuf::from(output_path)),
            force: false,
            skip_verification: false,
            strict_verification: false,
            manifest_dir: None,
            calm_config: None,
            autoencoder_config: None,
            tags: Vec::new(),
        }),
    };

    run_convert_command(args)?;

    println!("✓ Conversion complete: {}", output_path);
    Ok(())
}

fn encode_text(model_id: &str, text: &str) -> Result<()> {
    println!("Encoding with {}: \"{}\"", model_id, text);

    // 1. Load tokenizer
    println!("Loading tokenizer...");
    let tokenizer_loader = TokenizerLoader::new()?;
    let tokenizer = tokenizer_loader.load_from_hub(model_id, TokenizerType::Llama)?;

    // 2. Tokenize text
    println!("Tokenizing text...");
    let token_ids = TokenizerUtils::encode(&tokenizer, text, true)?;
    println!("Token IDs: {:?}", token_ids);

    // 3. Download model if needed and get config
    println!("Downloading model files...");
    let download = DownloadUtils::download_complete_model(model_id)?;

    // 4. Load autoencoder config
    let config = if let Some(config_path) = download.config_path() {
        // Try to load as CalmAutoencoderConfig first
        match CalmAutoencoderConfig::from_json_file(config_path) {
            Ok(config) => config,
            Err(_) => {
                // If that fails, try to load as core AutoencoderConfig and convert
                match AutoencoderConfig::from_json_file(config_path.to_str().unwrap()) {
                    Ok(core_config) => {
                        convert_autoencoder_config(&core_config)
                    }
                    Err(_) => {
                        // Finally fall back to generic JSON parsing
                        match std::fs::read_to_string(config_path) {
                            Ok(content) => {
                                if let Ok(json_value) = serde_json::from_str::<serde_json::Value>(&content) {
                                    convert_to_calm_autoencoder_config(&json_value)
                                } else {
                                    println!("Failed to parse config, using default");
                                    CalmAutoencoderConfig::default()
                                }
                            }
                            Err(_) => {
                                println!("Failed to read config file, using default");
                                CalmAutoencoderConfig::default()
                            }
                        }
                    }
                }
            }
        }
    } else {
        println!("No config found, using default CALM autoencoder config");
        CalmAutoencoderConfig::default()
    };

    // 5. Load autoencoder weights from safetensors
    println!("Loading autoencoder weights...");
    let device = auto_device()?;
    let safetensors_path = &download.model_paths()[0]; // Use first model file
    let autoencoder = CalmAutoencoder::from_safetensors(safetensors_path, config, DType::F32, &device)?;

    // 6. Create tensor from token IDs
    let seq_len = token_ids.len(); let input_ids = Tensor::from_vec(token_ids, (1, seq_len), &device)?;

    // 7. Run encoder to get pooled embedding
    println!("Running encoder...");
    let pooled_embedding = autoencoder.encode_pooled(&input_ids)?;

    // 8. Print the embedding
    println!("Pooled embedding shape: {:?}", pooled_embedding.dims());
    let embedding_values: Vec<f32> = pooled_embedding.to_vec1()?;
    println!("Pooled embedding values (first 10): {:?}", &embedding_values[..embedding_values.len().min(10)]);

    println!("✓ Encoding complete");
    Ok(())
}

/// Convert a generic JSON config to CalmAutoencoderConfig
fn convert_to_calm_autoencoder_config(json_value: &serde_json::Value) -> CalmAutoencoderConfig {
    let mut config = CalmAutoencoderConfig::default();

    // Extract common fields if they exist
    if let Some(vocab_size) = json_value.get("vocab_size").and_then(|v| v.as_u64()) {
        config.vocab_size = vocab_size as usize;
    }

    if let Some(hidden_size) = json_value.get("hidden_size").and_then(|v| v.as_u64()) {
        config.hidden_size = hidden_size as usize;
    }

    if let Some(intermediate_size) = json_value.get("intermediate_size").and_then(|v| v.as_u64()) {
        config.intermediate_size = intermediate_size as usize;
    }

    if let Some(patch_size) = json_value.get("patch_size").and_then(|v| v.as_u64()) {
        config.patch_size = patch_size as usize;
    }

    if let Some(latent_size) = json_value.get("latent_size").and_then(|v| v.as_u64()) {
        config.latent_size = latent_size as usize;
    }

    if let Some(num_encoder_layers) = json_value.get("num_encoder_layers").and_then(|v| v.as_u64()) {
        config.num_encoder_layers = num_encoder_layers as usize;
    }

    if let Some(num_decoder_layers) = json_value.get("num_decoder_layers").and_then(|v| v.as_u64()) {
        config.num_decoder_layers = num_decoder_layers as usize;
    }

    if let Some(ae_dropout) = json_value.get("ae_dropout").and_then(|v| v.as_f64()) {
        config.ae_dropout = ae_dropout;
    }

    if let Some(kl_clamp) = json_value.get("kl_clamp").and_then(|v| v.as_f64()) {
        config.kl_clamp = kl_clamp;
    }

    if let Some(kl_weight) = json_value.get("kl_weight").and_then(|v| v.as_f64()) {
        config.kl_weight = kl_weight;
    }

    if let Some(rms_norm_eps) = json_value.get("rms_norm_eps").and_then(|v| v.as_f64()) {
        config.rms_norm_eps = rms_norm_eps;
    }

    if let Some(bos_token_id) = json_value.get("bos_token_id").and_then(|v| v.as_u64()) {
        config.bos_token_id = bos_token_id as u32;
    }

    if let Some(eos_token_id) = json_value.get("eos_token_id").and_then(|v| v.as_u64()) {
        config.eos_token_id = eos_token_id as u32;
    }

    if let Some(pad_token_id) = json_value.get("pad_token_id").and_then(|v| v.as_u64()) {
        config.pad_token_id = Some(pad_token_id as u32);
    }

    config
}

fn score_text(model_id: &str, text: &str) -> Result<()> {
    println!("Scoring with {}: \"{}\"", model_id, text);

    // 1. Load tokenizer
    println!("Loading tokenizer...");
    let tokenizer_loader = TokenizerLoader::new()?;
    let tokenizer = tokenizer_loader.load_from_hub(model_id, TokenizerType::Llama)?;

    // 2. Tokenize text
    println!("Tokenizing text...");
    let token_ids = TokenizerUtils::encode(&tokenizer, text, true)?;
    println!("Token IDs: {:?}", token_ids);

    // 3. Download model files
    println!("Downloading model files...");
    let download = DownloadUtils::download_complete_model(model_id)?;

    // 4. Load autoencoder config and model
    let ae_config = if let Some(config_path) = download.config_path() {
        match CalmAutoencoderConfig::from_json_file(config_path) {
            Ok(config) => config,
            Err(_) => {
                match AutoencoderConfig::from_json_file(config_path.to_str().unwrap()) {
                    Ok(core_config) => convert_autoencoder_config(&core_config),
                    Err(_) => {
                        println!("Failed to parse config, using default");
                        CalmAutoencoderConfig::default()
                    }
                }
            }
        }
    } else {
        println!("No config found, using default CALM autoencoder config");
        CalmAutoencoderConfig::default()
    };

    // 5. Load device and autoencoder
    let device = auto_device()?;
    let safetensors_path = &download.model_paths()[0];
    let autoencoder = CalmAutoencoder::from_safetensors(safetensors_path, ae_config, DType::F32, &device)?;

    // 6. Create input tensor
    let seq_len = token_ids.len();
    let input_ids = Tensor::from_vec(token_ids, (1, seq_len), &device)?;

    // 7. Compute energy score using autoencoder latents
    println!("Computing energy score...");

    // Encode with autoencoder to get latent representation
    let latent_embedding = autoencoder.encode_pooled(&input_ids)?;

    // Simple energy score approximation using latent statistics
    let mean_norm = latent_embedding.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
    let variance = latent_embedding.var_keepdim(1)?.mean_all()?.to_scalar::<f32>()? as f64;

    // Simplified energy score based on latent statistics
    let energy_score = -(variance + mean_norm * 0.1);

    println!("✓ Energy score: {:.6}", energy_score);
    println!("✓ Latent mean norm: {:.6}", mean_norm);
    println!("✓ Latent variance: {:.6}", variance);

    // 8. Compute simple BrierLM scores for demonstration
    let brier_scores = (0.8f64, 0.7f64, 0.6f64, 0.5f64);
    let brier_lm = (brier_scores.0 * brier_scores.1 * brier_scores.2 * brier_scores.3).powf(0.25);
    println!("✓ BrierLM score (demo): {:.6}", brier_lm);

    println!("✓ Text scoring complete");
    Ok(())
}

/// Convert core AutoencoderConfig to CalmAutoencoderConfig
fn convert_autoencoder_config(core_config: &AutoencoderConfig) -> CalmAutoencoderConfig {
    CalmAutoencoderConfig {
        ae_dropout: core_config.ae_dropout,
        kl_clamp: core_config.kl_clamp,
        kl_weight: core_config.kl_weight,
        patch_size: core_config.patch_size as usize,
        vocab_size: core_config.vocab_size as usize,
        hidden_size: core_config.hidden_size as usize,
        intermediate_size: core_config.intermediate_size as usize,
        num_encoder_layers: core_config.num_encoder_layers as usize,
        num_decoder_layers: core_config.num_decoder_layers as usize,
        latent_size: core_config.latent_size as usize,
        hidden_act: core_config.hidden_act.clone(),
        max_position_embeddings: core_config.max_position_embeddings as usize,
        initializer_range: core_config.initializer_range,
        rms_norm_eps: core_config.rms_norm_eps,
        pad_token_id: core_config.pad_token_id,
        bos_token_id: core_config.bos_token_id,
        eos_token_id: core_config.eos_token_id,
        pretraining_tp: core_config.pretraining_tp as usize,
        tie_word_embeddings: core_config.tie_word_embeddings,
        mlp_bias: core_config.mlp_bias,
    }
}
