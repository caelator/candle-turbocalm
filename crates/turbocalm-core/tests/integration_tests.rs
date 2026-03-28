use anyhow::Result;
use turbocalm_core::*;

/// Integration tests for turbocalm-core functionality
mod integration {
    use super::*;

    #[test]
    fn test_calm_config_serialization_roundtrip() -> Result<()> {
        let original_config = CALMConfig {
            vocab_size: 50000,
            hidden_size: 1024,
            patch_size: 8,
            model_type: "test".to_string(),
            ..Default::default()
        };

        // Serialize to JSON
        let json = serde_json::to_string(&original_config)?;

        // Deserialize back
        let parsed_config: CALMConfig = serde_json::from_str(&json)?;

        // Verify all fields match
        assert_eq!(original_config.vocab_size, parsed_config.vocab_size);
        assert_eq!(original_config.hidden_size, parsed_config.hidden_size);
        assert_eq!(original_config.patch_size, parsed_config.patch_size);
        assert_eq!(original_config.model_type, parsed_config.model_type);

        Ok(())
    }

    #[test]
    fn test_autoencoder_config_serialization_roundtrip() -> Result<()> {
        let original_config = AutoencoderConfig {
            vocab_size: 30000,
            hidden_size: 256,
            ae_dropout: 0.2,
            kl_weight: 1e-4,
            ..Default::default()
        };

        let json = serde_json::to_string(&original_config)?;
        let parsed_config: AutoencoderConfig = serde_json::from_str(&json)?;

        assert_eq!(original_config.vocab_size, parsed_config.vocab_size);
        assert_eq!(original_config.hidden_size, parsed_config.hidden_size);
        assert!((original_config.ae_dropout - parsed_config.ae_dropout).abs() < 1e-10);
        assert!((original_config.kl_weight - parsed_config.kl_weight).abs() < 1e-10);

        Ok(())
    }

    #[test]
    fn test_device_selection_workflow() -> Result<()> {
        // Test automatic device selection
        let auto_device = auto_device();
        assert!(auto_device.is_ok());

        // Test cached device (should return the same device)
        let cached_1 = cached_device();
        let cached_2 = cached_device();

        assert!(cached_1.is_ok());
        assert!(cached_2.is_ok());

        // Both should point to the same cached device
        assert_eq!(
            cached_1.unwrap() as *const _,
            cached_2.unwrap() as *const _
        );

        Ok(())
    }

    #[test]
    fn test_device_info_and_selection() {
        let device_info = DeviceSelector::device_info();

        // CPU should always be available
        assert!(device_info.cpu_available);

        // Test device type checking
        assert!(DeviceSelector::is_available(DeviceType::Cpu));

        // Test best device selection
        let best_device = device_info.best_device_type();

        // Should be CPU if no GPU available, or GPU if available
        assert!(matches!(best_device, DeviceType::Cpu | DeviceType::Metal | DeviceType::Cuda));

        println!("{}", device_info);
    }

    #[test]
    fn test_error_handling_and_conversion() {
        // Test error conversions
        let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "test error");
        let turbo_error: TurboCALMError = io_error.into();

        assert!(matches!(turbo_error, TurboCALMError::Io(_)));

        // Test error macros
        let config_err = config_error!("test_param", "invalid value");
        assert!(matches!(config_err, TurboCALMError::Config(_)));

        let tensor_err = tensor_error!(shape_mismatch, "[2, 3]", "[3, 4]");
        assert!(matches!(tensor_err, TurboCALMError::Tensor(_)));
    }

    #[test]
    fn test_metrics_computation() -> Result<()> {
        use candle_core::{Device, DType, Tensor};

        // Create test tensors
        let device = Device::Cpu;
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0], (3,), &device)?;
        let b = Tensor::from_slice(&[1.0f32, 2.0, 3.0], (3,), &device)?;
        let c = Tensor::from_slice(&[2.0f32, 3.0, 4.0], (3,), &device)?;

        // Test cosine similarity
        let cos_sim_identical = SimilarityMetrics::cosine_similarity(&a, &b)?;
        assert!((cos_sim_identical - 1.0).abs() < 1e-6);

        let cos_sim_different = SimilarityMetrics::cosine_similarity(&a, &c)?;
        assert!(cos_sim_different > 0.9); // Should be high but not 1.0

        // Test MSE
        let mse_identical = SimilarityMetrics::mse(&a, &b)?;
        assert!(mse_identical < 1e-6);

        let mse_different = SimilarityMetrics::mse(&a, &c)?;
        assert!(mse_different > 0.0);

        // Test all metrics bundle
        let metrics = SimilarityMetrics::all_metrics(&a, &c)?;
        assert!(metrics.cosine_similarity > 0.9);
        assert!(metrics.mse > 0.0);
        assert!(metrics.rmse > 0.0);
        assert!(metrics.mae > 0.0);

        println!("Metrics: {}", metrics.summary());

        Ok(())
    }

    #[test]
    fn test_memory_reporting() {
        use candle_core::{Device, DType, Tensor};

        // Create a test tensor
        let tensor = Tensor::zeros((1000, 1000), DType::F32, &Device::Cpu).unwrap();

        // Get memory info
        let memory_info = MemoryReporter::tensor_memory_usage(&tensor);

        assert_eq!(memory_info.element_count, 1_000_000);
        assert_eq!(memory_info.size_bytes, 4_000_000); // 4 bytes per f32
        assert!((memory_info.size_mb - 3.8).abs() < 0.1); // ~3.8 MB

        println!("Tensor size: {}", memory_info.size_string());

        // Test memory usage reporting
        let usage = MemoryReporter::current_memory_usage();
        assert!(usage.current_mb >= 0.0);
        assert!(usage.available_mb >= 0.0);

        // Log memory (shouldn't crash)
        MemoryReporter::log_memory_usage();
        MemoryReporter::log_tensor_memory("test_tensor", &tensor);
    }

    #[test]
    fn test_metrics_tracker() {
        let mut tracker = MetricsTracker::new();

        // Record some metrics
        tracker.record("loss", 1.0);
        tracker.record("loss", 0.8);
        tracker.record("loss", 0.6);
        tracker.record("accuracy", 0.85);
        tracker.record("accuracy", 0.90);

        // Test retrieval
        assert_eq!(tracker.latest("loss"), Some(0.6));
        assert_eq!(tracker.latest("accuracy"), Some(0.90));

        let avg_loss = tracker.average("loss").unwrap();
        assert!((avg_loss - 0.8).abs() < 1e-6);

        let avg_acc = tracker.average("accuracy").unwrap();
        assert!((avg_acc - 0.875).abs() < 1e-6);

        // Test metric names
        let names = tracker.metric_names();
        assert!(names.contains(&"loss".to_string()));
        assert!(names.contains(&"accuracy".to_string()));

        // Test values
        assert_eq!(tracker.values("loss").unwrap().len(), 3);
        assert_eq!(tracker.values("accuracy").unwrap().len(), 2);

        // Test log summary (shouldn't crash)
        tracker.log_summary();

        // Test clear
        tracker.clear();
        assert!(tracker.metric_names().is_empty());
    }

    #[cfg(test)]
    mod config_validation_tests {
        use super::*;

        #[test]
        fn test_calm_config_validation() {
            let mut config = CALMConfig::default();

            // Valid config should pass
            assert!(config.validate().is_ok());

            // Invalid: zero attention heads
            config.num_attention_heads = 0;
            assert!(config.validate().is_err());

            // Invalid: hidden size not divisible by attention heads
            config.num_attention_heads = 13; // 768 not divisible by 13
            assert!(config.validate().is_err());

            // Valid: fix the config
            config.num_attention_heads = 12; // 768 divisible by 12
            assert!(config.validate().is_ok());
        }

        #[test]
        fn test_rope_scaling_validation() {
            let valid_rope = RopeScaling {
                rope_type: "linear".to_string(),
                factor: 2.0,
            };
            assert!(valid_rope.validate().is_ok());

            let invalid_type = RopeScaling {
                rope_type: "invalid".to_string(),
                factor: 2.0,
            };
            assert!(invalid_type.validate().is_err());

            let invalid_factor = RopeScaling {
                rope_type: "linear".to_string(),
                factor: 0.5, // Must be > 1.0
            };
            assert!(invalid_factor.validate().is_err());
        }

        #[test]
        fn test_config_num_key_value_heads() {
            let mut config = CALMConfig::default();
            config.num_attention_heads = 12;

            // Test default behavior (should equal attention heads)
            config.num_key_value_heads = None;
            assert_eq!(config.num_key_value_heads(), 12);

            // Test explicit value
            config.num_key_value_heads = Some(4);
            assert_eq!(config.num_key_value_heads(), 4);
        }
    }
}

/// Test module specifically for HuggingFace hub functionality
#[cfg(feature = "integration")]
mod hub_integration_tests {
    use super::*;
    use std::time::Duration;

    // These tests require network access and are disabled by default
    // Run with: cargo test --features integration

    #[tokio::test]
    #[ignore] // Ignored by default, can be run with --ignored
    async fn test_hub_client_creation() -> Result<()> {
        let client = HubClient::new();
        assert!(client.is_ok());
        Ok(())
    }

    #[tokio::test]
    #[ignore]
    async fn test_model_existence_check() -> Result<()> {
        let client = HubClient::new()?;

        // Test with a known public model (should exist)
        let exists = client.model_exists("bert-base-uncased");
        // Note: This might fail if there's no network connection

        println!("bert-base-uncased exists: {}", exists);
        Ok(())
    }

    #[test]
    #[ignore]
    fn test_download_manifest_operations() -> Result<()> {
        use tempfile::TempDir;

        let temp_dir = TempDir::new()?;
        let manifest_path = temp_dir.path().join("test_manifest.json");

        // Create a test manifest
        let mut manifest = DownloadManifest::new("test/model");
        manifest.add_file("config.json", temp_dir.path());

        // Save and load
        manifest.save_to_file(&manifest_path)?;
        let loaded_manifest = DownloadManifest::load_from_file(&manifest_path)?;

        assert_eq!(manifest.model_id, loaded_manifest.model_id);
        assert_eq!(manifest.files.len(), loaded_manifest.files.len());

        Ok(())
    }
}

/// Stress tests for performance validation
mod stress_tests {
    use super::*;

    #[test]
    fn test_large_tensor_metrics() -> Result<()> {
        use candle_core::{Device, DType, Tensor};

        // Create moderately large tensors
        let device = Device::Cpu;
        let size = 1000; // 1M elements
        let a = Tensor::zeros((size, size), DType::F32, &device)?;
        let b = Tensor::ones((size, size), DType::F32, &device)?;

        // This should complete without running out of memory
        let start = std::time::Instant::now();

        let mse = SimilarityMetrics::mse(&a, &b)?;
        let cos_sim = SimilarityMetrics::cosine_similarity(&a, &b)?;

        let elapsed = start.elapsed();

        println!("Large tensor metrics computed in: {:?}", elapsed);
        println!("MSE: {}, Cosine similarity: {}", mse, cos_sim);

        assert_eq!(mse, 1.0); // zeros vs ones should have MSE of 1
        assert_eq!(cos_sim, 0.0); // zero tensor has no meaningful cosine similarity

        Ok(())
    }

    #[test]
    fn test_metrics_tracker_performance() {
        let mut tracker = MetricsTracker::new();

        // Add many metrics
        let start = std::time::Instant::now();

        for i in 0..10000 {
            tracker.record("metric_1", i as f32);
            tracker.record("metric_2", (i as f32) * 0.5);
            tracker.record("metric_3", (i as f32).sin());
        }

        let elapsed = start.elapsed();
        println!("Added 30k metrics in: {:?}", elapsed);

        // Operations should still be fast
        let avg1 = tracker.average("metric_1").unwrap();
        let avg2 = tracker.average("metric_2").unwrap();
        let avg3 = tracker.average("metric_3").unwrap();

        println!("Averages: {}, {}, {}", avg1, avg2, avg3);

        assert!(elapsed.as_millis() < 1000); // Should complete in under 1 second
    }
}