use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use candle_core::{Device, Var};
use candle_nn::VarMap;
use turbocalm_models::CalmAutoencoderConfig;

const CHECKPOINT_PREFIX: &str = "checkpoint-v";
const CHECKPOINT_SUFFIX: &str = ".safetensors";
const CHECKPOINT_CONFIG_SUFFIX: &str = ".config.json";
const LATEST_CHECKPOINT_NAME: &str = "latest.safetensors";
const LATEST_CONFIG_NAME: &str = "latest-config.json";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointInfo {
    pub version: u64,
    pub path: PathBuf,
    pub size_bytes: u64,
    pub modified_unix_secs: Option<u64>,
}

pub fn default_checkpoint_dir() -> Result<PathBuf> {
    let home = std::env::var_os("HOME").context("HOME is not set")?;
    Ok(PathBuf::from(home).join(".turbocalm").join("trained"))
}

pub fn checkpoint_path_for_version(version: u64) -> Result<PathBuf> {
    checkpoint_path_in_dir(&default_checkpoint_dir()?, version)
}

pub fn checkpoint_path_in_dir(dir: &Path, version: u64) -> Result<PathBuf> {
    Ok(dir.join(checkpoint_file_name(version)))
}

pub fn latest_checkpoint_path() -> Result<PathBuf> {
    Ok(latest_checkpoint_path_in_dir(&default_checkpoint_dir()?))
}

pub fn latest_checkpoint_path_in_dir(dir: &Path) -> PathBuf {
    dir.join(LATEST_CHECKPOINT_NAME)
}

pub fn next_checkpoint_version() -> Result<u64> {
    next_checkpoint_version_in_dir(&default_checkpoint_dir()?)
}

pub fn next_checkpoint_version_in_dir(dir: &Path) -> Result<u64> {
    let mut max_version = 0u64;
    if dir.exists() {
        for entry in std::fs::read_dir(dir)
            .with_context(|| format!("failed to read checkpoint dir {}", dir.display()))?
        {
            let entry = entry.with_context(|| format!("failed to read {}", dir.display()))?;
            if let Some(version) = checkpoint_version_from_path(&entry.path()) {
                max_version = max_version.max(version);
            }
        }
    }
    Ok(max_version + 1)
}

pub fn save_checkpoint<P: AsRef<Path>>(
    varmap: &VarMap,
    path: P,
    version: u64,
) -> Result<CheckpointInfo> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }

    let temp_path = temp_path_for(path);
    varmap
        .save(&temp_path)
        .with_context(|| format!("failed to save checkpoint to {}", temp_path.display()))?;
    load_checkpoint(&temp_path)
        .with_context(|| format!("failed to verify checkpoint {}", temp_path.display()))?;
    fs::rename(&temp_path, path).with_context(|| {
        format!(
            "failed to atomically install checkpoint {}",
            path.display()
        )
    })?;

    if let Some(parent) = path.parent() {
        let latest_path = latest_checkpoint_path_in_dir(parent);
        promote_file_atomically(path, &latest_path)?;
    }
    checkpoint_info(path.to_path_buf(), version)
}

pub fn load_checkpoint<P: AsRef<Path>>(path: P) -> Result<VarMap> {
    let path = path.as_ref();
    let tensors = candle_core::safetensors::load(path, &Device::Cpu)
        .with_context(|| format!("failed to load checkpoint {}", path.display()))?;
    let varmap = VarMap::new();
    let mut data = varmap.data().lock().unwrap();
    for (name, tensor) in tensors {
        data.insert(name, Var::from_tensor(&tensor).context("failed to restore variable")?);
    }
    drop(data);
    Ok(varmap)
}

pub fn list_checkpoints() -> Result<Vec<CheckpointInfo>> {
    list_checkpoints_in_dir(&default_checkpoint_dir()?)
}

pub fn list_checkpoints_in_dir(dir: &Path) -> Result<Vec<CheckpointInfo>> {
    if !dir.exists() {
        return Ok(Vec::new());
    }

    let mut checkpoints = Vec::new();
    for entry in std::fs::read_dir(dir)
        .with_context(|| format!("failed to read checkpoint dir {}", dir.display()))?
    {
        let entry = entry.with_context(|| format!("failed to read {}", dir.display()))?;
        let path = entry.path();
        let Some(version) = checkpoint_version_from_path(&path) else {
            continue;
        };
        checkpoints.push(checkpoint_info(path, version)?);
    }
    checkpoints.sort_by_key(|checkpoint| checkpoint.version);
    Ok(checkpoints)
}

pub fn save_checkpoint_config<P: AsRef<Path>>(
    config: &CalmAutoencoderConfig,
    checkpoint_path: P,
) -> Result<PathBuf> {
    let checkpoint_path = checkpoint_path.as_ref();
    let config_path = config_path_for_checkpoint(checkpoint_path);
    save_config_atomically(config, &config_path)?;

    if let Some(parent) = checkpoint_path.parent() {
        let latest_config_path = parent.join(LATEST_CONFIG_NAME);
        save_config_atomically(config, &latest_config_path)?;
    }

    Ok(config_path)
}

pub fn load_checkpoint_config<P: AsRef<Path>>(checkpoint_path: P) -> Result<Option<CalmAutoencoderConfig>> {
    let checkpoint_path = checkpoint_path.as_ref();
    let candidate_paths = checkpoint_config_candidates(checkpoint_path);
    for candidate in candidate_paths {
        if !candidate.exists() {
            continue;
        }
        let raw = fs::read_to_string(&candidate)
            .with_context(|| format!("failed to read checkpoint config {}", candidate.display()))?;
        let config = serde_json::from_str::<CalmAutoencoderConfig>(&raw).with_context(|| {
            format!("failed to parse checkpoint config {}", candidate.display())
        })?;
        return Ok(Some(config));
    }
    Ok(None)
}

fn checkpoint_info(path: PathBuf, version: u64) -> Result<CheckpointInfo> {
    let metadata = fs::metadata(&path)
        .with_context(|| format!("failed to read metadata for {}", path.display()))?;
    Ok(CheckpointInfo {
        version,
        path,
        size_bytes: metadata.len(),
        modified_unix_secs: metadata
            .modified()
            .ok()
            .and_then(system_time_to_unix_secs),
    })
}

fn system_time_to_unix_secs(time: SystemTime) -> Option<u64> {
    time.duration_since(UNIX_EPOCH).ok().map(|duration| duration.as_secs())
}

fn checkpoint_file_name(version: u64) -> String {
    format!("{CHECKPOINT_PREFIX}{version:06}{CHECKPOINT_SUFFIX}")
}

fn config_path_for_checkpoint(path: &Path) -> PathBuf {
    let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
        return path.with_extension(CHECKPOINT_CONFIG_SUFFIX.trim_start_matches('.'));
    };

    if file_name == LATEST_CHECKPOINT_NAME {
        return path
            .parent()
            .map(|parent| parent.join(LATEST_CONFIG_NAME))
            .unwrap_or_else(|| PathBuf::from(LATEST_CONFIG_NAME));
    }

    let config_name = format!(
        "{}{}",
        file_name.trim_end_matches(CHECKPOINT_SUFFIX),
        CHECKPOINT_CONFIG_SUFFIX
    );
    path.parent()
        .map(|parent| parent.join(&config_name))
        .unwrap_or_else(|| PathBuf::from(config_name))
}

fn checkpoint_version_from_path(path: &Path) -> Option<u64> {
    let name = path.file_name()?.to_str()?;
    let digits = name
        .strip_prefix(CHECKPOINT_PREFIX)?
        .strip_suffix(CHECKPOINT_SUFFIX)?;
    digits.parse::<u64>().ok()
}

fn checkpoint_config_candidates(path: &Path) -> Vec<PathBuf> {
    let mut candidates = vec![config_path_for_checkpoint(path)];
    if let Some(parent) = path.parent() {
        let latest = parent.join(LATEST_CONFIG_NAME);
        if !candidates.contains(&latest) {
            candidates.push(latest);
        }
    }
    candidates
}

fn save_config_atomically(config: &CalmAutoencoderConfig, path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let temp_path = temp_path_for(path);
    let raw = serde_json::to_string_pretty(config).context("failed to serialize checkpoint config")?;
    fs::write(&temp_path, raw.as_bytes())
        .with_context(|| format!("failed to write checkpoint config {}", temp_path.display()))?;
    let verified = fs::read_to_string(&temp_path)
        .with_context(|| format!("failed to verify config {}", temp_path.display()))?;
    serde_json::from_str::<CalmAutoencoderConfig>(&verified)
        .with_context(|| format!("failed to parse config {}", temp_path.display()))?;
    fs::rename(&temp_path, path)
        .with_context(|| format!("failed to atomically install {}", path.display()))?;
    Ok(())
}

fn promote_file_atomically(source: &Path, destination: &Path) -> Result<()> {
    let temp_path = temp_path_for(destination);
    fs::copy(source, &temp_path).with_context(|| {
        format!(
            "failed to stage checkpoint copy from {} to {}",
            source.display(),
            temp_path.display()
        )
    })?;
    load_checkpoint(&temp_path)
        .with_context(|| format!("failed to verify staged checkpoint {}", temp_path.display()))?;
    fs::rename(&temp_path, destination).with_context(|| {
        format!(
            "failed to atomically promote checkpoint {}",
            destination.display()
        )
    })?;
    Ok(())
}

fn temp_path_for(path: &Path) -> PathBuf {
    let suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("checkpoint");
    path.parent()
        .map(|parent| parent.join(format!("{file_name}.tmp-{suffix}")))
        .unwrap_or_else(|| PathBuf::from(format!("{file_name}.tmp-{suffix}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::{Init, VarBuilder};

    #[test]
    fn round_trips_varmap_from_checkpoint() -> Result<()> {
        let temp_root = std::env::temp_dir().join(format!(
            "turbocalm-train-checkpoint-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("unnamed")
        ));
        std::fs::create_dir_all(&temp_root)?;
        let checkpoint_path = temp_root.join(checkpoint_file_name(1));

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        vb.get_with_hints((2, 2), "weight", Init::Const(1.0))?;

        save_checkpoint(&varmap, &checkpoint_path, 1)?;
        let loaded = load_checkpoint(&checkpoint_path)?;
        let original = varmap
            .data()
            .lock()
            .unwrap()
            .get("weight")
            .unwrap()
            .as_tensor()
            .flatten_all()?
            .to_vec1::<f32>()?;
        let restored = loaded
            .data()
            .lock()
            .unwrap()
            .get("weight")
            .unwrap()
            .as_tensor()
            .flatten_all()?
            .to_vec1::<f32>()?;

        assert_eq!(original, restored);
        std::fs::remove_file(checkpoint_path)?;
        std::fs::remove_dir_all(temp_root)?;
        Ok(())
    }

    #[test]
    fn saves_latest_checkpoint_and_config_atomically() -> Result<()> {
        let temp_root = std::env::temp_dir().join(format!(
            "turbocalm-train-checkpoint-meta-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        fs::create_dir_all(&temp_root)?;

        let checkpoint_path = temp_root.join(checkpoint_file_name(3));
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        vb.get_with_hints((2, 2), "weight", Init::Const(2.0))?;

        save_checkpoint(&varmap, &checkpoint_path, 3)?;
        let latest_path = latest_checkpoint_path_in_dir(&temp_root);
        assert!(latest_path.exists());
        assert!(load_checkpoint(&latest_path).is_ok());

        let config = CalmAutoencoderConfig::default();
        save_checkpoint_config(&config, &checkpoint_path)?;
        let loaded_config = load_checkpoint_config(&checkpoint_path)?.unwrap();
        assert_eq!(loaded_config, config);

        fs::remove_dir_all(temp_root)?;
        Ok(())
    }
}
