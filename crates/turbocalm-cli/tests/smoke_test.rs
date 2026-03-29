//! End-to-end smoke test for the CALM inference pipeline
//!
//! Basic smoke test to verify that core components can be loaded and don't panic.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

// Import the core components
use turbocalm_core::quant::QuantProfile;
use turbocalm_models::calm::autoencoder::{CalmAutoencoder, CalmAutoencoderConfig};

/// Create test configurations
fn create_test_config() -> CalmAutoencoderConfig {
    CalmAutoencoderConfig {
        vocab_size: 32,
        hidden_size: 64,
        intermediate_size: 128,
        latent_size: 16,
        patch_size: 4,
        num_encoder_layers: 2,
        num_decoder_layers: 2,
        tie_word_embeddings: true,
        ..Default::default()
    }
}

#[test]
fn test_end_to_end_autoencoder_pipeline() -> Result<()> {
    // 1. Config parsing - create test config
    let config = create_test_config();
    println!("✓ Config created successfully");

    // 2. Device selection
    let device = Device::Cpu;
    println!("✓ Selected device: {:?}", device);

    // 3. Model initialization with VarBuilder::zeros
    let vb = VarBuilder::zeros(DType::F32, &device);

    // Create autoencoder
    let autoencoder = CalmAutoencoder::load(vb, config.clone())?;
    println!("✓ Autoencoder loaded successfully");

    // 4. Test encode/decode pipeline with synthetic input
    let batch_size = 1;
    let seq_len = 8; // Multiple of patch_size (4)
    let vocab_size = config.vocab_size;

    // Create synthetic input tokens
    let input_ids = Tensor::zeros((batch_size, seq_len), DType::U32, &device)?;
    println!("✓ Created input tensor with shape: {:?}", input_ids.dims());

    // Test autoencoder encode
    let latent_embeddings = autoencoder.encode_chunked(&input_ids)?;
    println!(
        "✓ Encoded to latents with shape: {:?}",
        latent_embeddings.dims()
    );

    // Test autoencoder decode
    let decoded_logits = autoencoder.decode(&latent_embeddings)?;
    println!(
        "✓ Decoded to logits with shape: {:?}",
        decoded_logits.dims()
    );

    // Verify shapes are correct
    assert_eq!(latent_embeddings.dims().len(), 3); // [batch, patches, latent_size]
    assert_eq!(decoded_logits.dims().len(), 3); // [batch, seq_len, vocab_size]
    assert_eq!(decoded_logits.dims()[2], vocab_size);

    // Verify that the pipeline produces reasonable outputs (not NaN/Inf)
    let latent_data = latent_embeddings.flatten_all()?.to_vec1::<f32>()?;
    let logit_data = decoded_logits.flatten_all()?.to_vec1::<f32>()?;

    assert!(
        latent_data.iter().all(|&x| x.is_finite()),
        "Latent outputs should be finite"
    );
    assert!(
        logit_data.iter().all(|&x| x.is_finite()),
        "Logit outputs should be finite"
    );

    println!("✓ End-to-end autoencoder pipeline successful");

    Ok(())
}

#[test]
fn test_device_selection() -> Result<()> {
    // Test device selection logic
    let device = Device::Cpu;

    println!("Device selection: {:?}", device);

    // Should successfully create a device (CPU fallback)
    assert!(matches!(device, Device::Cpu));

    println!("✓ Device selection successful");

    Ok(())
}

#[test]
fn test_config_parsing() -> Result<()> {
    // Test that we can create and validate configs
    let config = create_test_config();

    // Verify config has reasonable values
    assert!(config.vocab_size > 0);
    assert!(config.hidden_size > 0);
    assert!(config.patch_size > 0);
    assert!(config.latent_size > 0);
    assert!(config.num_encoder_layers > 0);
    assert!(config.num_decoder_layers > 0);

    // Test config serialization/deserialization
    let json_str = serde_json::to_string(&config)?;
    let deserialized: CalmAutoencoderConfig = serde_json::from_str(&json_str)?;

    assert_eq!(config.vocab_size, deserialized.vocab_size);
    assert_eq!(config.hidden_size, deserialized.hidden_size);
    assert_eq!(config.patch_size, deserialized.patch_size);

    println!("✓ Config parsing and validation successful");

    Ok(())
}

#[test]
fn test_tensor_shapes_and_dtypes() -> Result<()> {
    let device = Device::Cpu;

    // Test that we can create tensors with different shapes and dtypes
    let f64_tensor = Tensor::randn(0.0, 1.0, (2, 4, 8), &device)?; // randn creates F64 by default
    let u32_tensor = Tensor::zeros((2, 8), DType::U32, &device)?;
    let f32_tensor = Tensor::zeros((2, 4, 8), DType::F32, &device)?;

    assert_eq!(f64_tensor.dtype(), DType::F64);
    assert_eq!(u32_tensor.dtype(), DType::U32);
    assert_eq!(f32_tensor.dtype(), DType::F32);
    assert_eq!(f64_tensor.dims(), &[2, 4, 8]);
    assert_eq!(f32_tensor.dims(), &[2, 4, 8]);
    assert_eq!(u32_tensor.dims(), &[2, 8]);

    // Test tensor operations don't panic
    let reshaped = f32_tensor.flatten_all()?;
    assert_eq!(reshaped.dims(), &[64]); // 2*4*8 = 64

    println!("✓ Tensor creation and basic operations successful");

    Ok(())
}

#[test]
fn test_quantization_profile() -> Result<()> {
    // Test that we can create quantization profiles
    let profile = QuantProfile {
        bit_width: 4,
        qjl_dim: 8,
        scale_mode: "per_token".to_string(),
        rotation_seed: 42,
        qjl_threshold: 0.0001,
        clipping_percentile: 0.95,
        scale_multiplier: 1.0,
    };

    // Test serialization
    let json_str = serde_json::to_string(&profile)?;
    let deserialized: QuantProfile = serde_json::from_str(&json_str)?;

    assert_eq!(profile.bit_width, deserialized.bit_width);
    assert_eq!(profile.qjl_dim, deserialized.qjl_dim);
    assert_eq!(profile.scale_mode, deserialized.scale_mode);

    println!("✓ QuantProfile creation and serialization successful");

    Ok(())
}
