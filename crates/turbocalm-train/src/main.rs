use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use candle_core::Device;
use clap::{Parser, Subcommand, ValueEnum};
use turbocalm_models::CalmAutoencoderConfig;
use turbocalm_train::server::{resolve_mode, ServerState};
use turbocalm_train::{
    checkpoint, corpus, run_eval, serve, spike, EmbeddingModel, EvalCorpus, Trainer,
    TrainingConfig, DEFAULT_MODEL_NAME,
};

#[derive(Parser)]
#[command(name = "turbocalm-train", about = "TurboCALM training utilities")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run the original forward/backward/optimizer spike.
    Spike,
    /// Train the CALM autoencoder on a JSONL corpus.
    Train {
        #[arg(long)]
        corpus: PathBuf,
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long, value_enum, default_value_t = DeviceArg::Auto)]
        device: DeviceArg,
        #[arg(long, default_value_t = 32)]
        batch_size: usize,
        #[arg(long, default_value_t = 1e-4)]
        lr: f64,
        #[arg(long, default_value_t = 0.01)]
        weight_decay: f64,
        #[arg(long, default_value_t = 0.07)]
        temperature: f64,
        #[arg(long, default_value_t = 20)]
        epochs: usize,
        #[arg(long, default_value_t = 2)]
        eval_interval: usize,
        #[arg(long, default_value_t = 3)]
        patience: usize,
        #[arg(long)]
        checkpoint_dir: Option<PathBuf>,
        #[arg(long, default_value_t = 200)]
        min_corpus_size: usize,
    },
    /// Evaluate a checkpoint against an eval corpus JSON file.
    Eval {
        #[arg(long)]
        corpus: PathBuf,
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long, value_enum, default_value_t = DeviceArg::Auto)]
        device: DeviceArg,
    },
    /// Serve OpenAI-compatible embeddings over HTTP.
    Serve {
        #[arg(long, default_value_t = 11435)]
        port: u16,
        #[arg(long)]
        checkpoint: Option<PathBuf>,
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long, value_enum, default_value_t = DeviceArg::Auto)]
        device: DeviceArg,
        #[arg(long, default_value_t = false, conflicts_with = "chunked")]
        pooled: bool,
        #[arg(long, default_value_t = false, conflicts_with = "pooled")]
        chunked: bool,
    },
    /// Corpus utilities.
    Corpus {
        #[command(subcommand)]
        command: CorpusCommands,
    },
    /// Checkpoint utilities.
    Checkpoints {
        #[command(subcommand)]
        command: CheckpointCommands,
    },
}

#[derive(Subcommand)]
enum CorpusCommands {
    /// Merge one or more JSONL corpora into a single output file.
    Build {
        #[arg(long, num_args = 1..)]
        sources: Vec<PathBuf>,
        #[arg(long)]
        output: PathBuf,
    },
}

#[derive(Subcommand)]
enum CheckpointCommands {
    /// List available checkpoints.
    List {
        #[arg(long)]
        dir: Option<PathBuf>,
    },
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum DeviceArg {
    Auto,
    Cpu,
    Metal,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Spike => {
            let report = spike::run_and_write_results()?;
            println!("{}", report.render_console());
        }
        Commands::Train {
            corpus,
            config,
            device,
            batch_size,
            lr,
            weight_decay,
            temperature,
            epochs,
            eval_interval,
            patience,
            checkpoint_dir,
            min_corpus_size,
        } => handle_train(
            &corpus,
            config.as_deref(),
            device,
            batch_size,
            lr,
            weight_decay,
            temperature,
            epochs,
            eval_interval,
            patience,
            checkpoint_dir,
            min_corpus_size,
        )?,
        Commands::Eval {
            corpus,
            checkpoint,
            config,
            device,
        } => handle_eval(&corpus, &checkpoint, config.as_deref(), device)?,
        Commands::Serve {
            port,
            checkpoint,
            config,
            device,
            pooled,
            chunked,
        } => handle_serve(port, checkpoint.as_deref(), config.as_deref(), device, pooled, chunked)
            .await?,
        Commands::Corpus { command } => match command {
            CorpusCommands::Build { sources, output } => handle_corpus_build(&sources, &output)?,
        },
        Commands::Checkpoints { command } => match command {
            CheckpointCommands::List { dir } => handle_checkpoint_list(dir.as_deref())?,
        },
    }
    Ok(())
}

fn handle_train(
    corpus_path: &Path,
    config_path: Option<&Path>,
    device: DeviceArg,
    batch_size: usize,
    lr: f64,
    weight_decay: f64,
    temperature: f64,
    epochs: usize,
    eval_interval: usize,
    patience: usize,
    checkpoint_dir: Option<PathBuf>,
    min_corpus_size: usize,
) -> Result<()> {
    let entries = corpus::load_from_jsonl(corpus_path)
        .with_context(|| format!("failed to load {}", corpus_path.display()))?;
    let pairs = corpus::build_pairs_from_entries(&entries);
    if pairs.pairs.is_empty() {
        bail!(
            "no training pairs were generated from {}",
            corpus_path.display()
        );
    }

    let model_config = load_model_config(config_path, None)?;
    let training_config = TrainingConfig {
        batch_size,
        lr,
        weight_decay,
        temperature,
        max_epochs: epochs,
        eval_interval,
        patience,
        checkpoint_dir: match checkpoint_dir {
            Some(dir) => dir,
            None => checkpoint::default_checkpoint_dir()?,
        },
        min_corpus_size,
    };

    println!(
        "loaded {} entries and generated {} training pairs",
        entries.len(),
        pairs.pairs.len()
    );

    let mut trainer = Trainer::new(model_config, training_config, resolve_device(device)?)?;
    let summary = trainer.train(&pairs)?;

    println!("best_loss={:.6}", summary.best_loss);
    if let Some(best_checkpoint) = summary.best_checkpoint {
        println!(
            "checkpoint=v{} path={}",
            best_checkpoint.version,
            best_checkpoint.path.display()
        );
        if let Some(parent) = best_checkpoint.path.parent() {
            println!(
                "latest={}",
                checkpoint::latest_checkpoint_path_in_dir(parent).display()
            );
        }
    }
    println!("epochs_ran={}", summary.epoch_losses.len());
    println!("stopped_early={}", summary.stopped_early);
    Ok(())
}

fn handle_eval(
    corpus_path: &Path,
    checkpoint_path: &Path,
    config_path: Option<&Path>,
    device: DeviceArg,
) -> Result<()> {
    let corpus = EvalCorpus::from_json_file(corpus_path)?;
    let config = load_model_config(config_path, Some(checkpoint_path))?;
    let model = EmbeddingModel::from_checkpoint(checkpoint_path, config, resolve_device(device)?)?;
    let metrics = run_eval(&model, &corpus)?;
    println!("{}", metrics.render_table());
    Ok(())
}

async fn handle_serve(
    port: u16,
    checkpoint_path: Option<&Path>,
    config_path: Option<&Path>,
    device: DeviceArg,
    pooled: bool,
    chunked: bool,
) -> Result<()> {
    let mode = resolve_mode(pooled, chunked)?;
    let checkpoint_path = checkpoint_path
        .map(PathBuf::from)
        .or_else(|| checkpoint::latest_checkpoint_path().ok())
        .filter(|path| path.exists());
    let config = load_model_config(config_path, checkpoint_path.as_deref())?;
    let device = resolve_device(device)?;

    let model = match checkpoint_path.as_deref() {
        Some(path) => {
            println!("loading checkpoint {}", path.display());
            match EmbeddingModel::from_checkpoint(path, config.clone(), device.clone()) {
                Ok(model) => model,
                Err(error) => {
                    println!(
                        "failed to load checkpoint {}; falling back to random weights: {error:#}",
                        path.display()
                    );
                    EmbeddingModel::random(config.clone(), device.clone())?
                }
            }
        }
        None => {
            println!("no checkpoint found; starting with random weights");
            EmbeddingModel::random(config.clone(), device.clone())?
        }
    };

    let state = ServerState {
        model: Arc::new(model),
        mode,
        model_name: Arc::from(DEFAULT_MODEL_NAME),
    };

    serve(state, port).await
}

fn handle_corpus_build(sources: &[PathBuf], output: &Path) -> Result<()> {
    if sources.is_empty() {
        bail!("--sources requires at least one JSONL path")
    }

    let mut merged = Vec::new();
    for source in sources {
        let mut entries = corpus::load_from_jsonl(source)
            .with_context(|| format!("failed to load {}", source.display()))?;
        merged.append(&mut entries);
    }

    corpus::save_to_jsonl(&merged, output)
        .with_context(|| format!("failed to write {}", output.display()))?;
    println!("merged {} entries into {}", merged.len(), output.display());
    Ok(())
}

fn handle_checkpoint_list(dir: Option<&Path>) -> Result<()> {
    let dir = match dir {
        Some(dir) => dir.to_path_buf(),
        None => checkpoint::default_checkpoint_dir()?,
    };
    let checkpoints = checkpoint::list_checkpoints_in_dir(&dir)?;
    if checkpoints.is_empty() {
        println!("no checkpoints found in {}", dir.display());
    } else {
        for checkpoint in checkpoints {
            println!(
                "v{} size={} modified={} path={}",
                checkpoint.version,
                checkpoint.size_bytes,
                checkpoint
                    .modified_unix_secs
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "unknown".to_string()),
                checkpoint.path.display()
            );
        }
    }
    Ok(())
}

fn load_model_config(
    config_path: Option<&Path>,
    checkpoint_path: Option<&Path>,
) -> Result<CalmAutoencoderConfig> {
    if let Some(path) = config_path {
        return CalmAutoencoderConfig::from_json_file(path)
            .with_context(|| format!("failed to load {}", path.display()));
    }

    if let Some(path) = checkpoint_path {
        if let Some(config) = checkpoint::load_checkpoint_config(path)? {
            return Ok(config);
        }
    }

    Ok(CalmAutoencoderConfig::default())
}

fn resolve_device(device: DeviceArg) -> Result<Device> {
    match device {
        DeviceArg::Auto => turbocalm_core::auto_device().context("failed to resolve auto device"),
        DeviceArg::Cpu => Ok(Device::Cpu),
        DeviceArg::Metal => Device::new_metal(0).context("failed to create Metal device"),
    }
}
