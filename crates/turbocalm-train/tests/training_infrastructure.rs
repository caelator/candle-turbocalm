use std::collections::BTreeMap;

use anyhow::Result;
use candle_core::{Device, Var};
use turbocalm_models::CalmAutoencoderConfig;
use turbocalm_train::{
    checkpoint::{load_checkpoint, save_checkpoint},
    pairs::{Corpus, CorpusMetadata, TrainingPair},
    Trainer, TrainingConfig,
};

#[test]
fn training_decreases_loss_and_round_trips_checkpoint() -> Result<()> {
    let temp_root = std::env::temp_dir().join(format!(
        "turbocalm-train-it-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    std::fs::create_dir_all(&temp_root)?;

    let model_config = CalmAutoencoderConfig {
        vocab_size: 512,
        hidden_size: 16,
        intermediate_size: 32,
        latent_size: 8,
        num_encoder_layers: 2,
        num_decoder_layers: 2,
        patch_size: 4,
        tie_word_embeddings: true,
        ..Default::default()
    };

    let training_config = TrainingConfig {
        batch_size: 20,
        lr: 0.01,
        weight_decay: 0.0,
        temperature: 0.07,
        max_epochs: 10,
        eval_interval: 1,
        patience: 10,
        checkpoint_dir: temp_root.clone(),
        min_corpus_size: 1,
    };

    let corpus = synthetic_corpus();
    let mut trainer = Trainer::new(model_config, training_config, Device::Cpu)?;

    let mut losses = Vec::new();
    for _ in 0..10 {
        losses.push(trainer.train_epoch(&corpus)?);
    }

    // With 10 epochs and lower lr, the best loss should beat the first
    let best = losses.iter().copied().fold(f32::INFINITY, f32::min);
    assert!(
        best < losses[0],
        "expected best loss ({best}) to beat epoch 1 ({})",
        losses[0]
    );
    assert!(losses.iter().all(|l| l.is_finite()));

    let checkpoint_path = temp_root.join("checkpoint-v000001.safetensors");
    save_checkpoint(&trainer.varmap, &checkpoint_path, 1)?;
    let restored = load_checkpoint(&checkpoint_path)?;

    assert_eq!(
        snapshot_all_vars(&trainer.varmap)?,
        snapshot_all_vars(&restored)?
    );

    std::fs::remove_file(checkpoint_path)?;
    std::fs::remove_dir_all(temp_root)?;
    Ok(())
}

fn synthetic_corpus() -> Corpus {
    let categories = [
        ("alpha", "breathing focus"),
        ("beta", "system design"),
        ("gamma", "daily planning"),
        ("delta", "debug tracing"),
    ];

    let mut pairs = Vec::new();
    for (category, phrase) in categories {
        for idx in 0..5 {
            pairs.push(TrainingPair {
                anchor: format!("{category} anchor {idx} {phrase} calm practice"),
                positive: format!("{category} positive {idx} {phrase} aligned memory"),
            });
        }
    }

    Corpus {
        metadata: CorpusMetadata {
            pair_count: pairs.len(),
            category_count: 4,
            source_count: 1,
            categorized_pair_count: pairs.len(),
            temporal_pair_count: 0,
        },
        pairs,
    }
}

fn snapshot_all_vars(varmap: &candle_nn::VarMap) -> Result<BTreeMap<String, Vec<f32>>> {
    let data = varmap.data().lock().unwrap();
    let mut names = data.keys().cloned().collect::<Vec<_>>();
    names.sort();

    let mut snapshots = BTreeMap::new();
    for name in names {
        let values = snapshot_var(data.get(&name).unwrap())?;
        snapshots.insert(name, values);
    }
    Ok(snapshots)
}

fn snapshot_var(var: &Var) -> Result<Vec<f32>> {
    var.as_tensor()
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()
        .map_err(Into::into)
}
