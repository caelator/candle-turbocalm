use candle_core::Device;
use std::sync::OnceLock;
use tracing::{info, warn};

/// Global device cache to avoid repeated device selection
static DEVICE_CACHE: OnceLock<Device> = OnceLock::new();

/// Device selection preferences
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DevicePreference {
    /// Prefer GPU acceleration if available, fallback to CPU
    Auto,
    /// Force CPU usage
    Cpu,
    /// Force GPU usage (Metal on macOS, CUDA elsewhere)
    Gpu,
}

impl Default for DevicePreference {
    fn default() -> Self {
        Self::Auto
    }
}

/// Device selection utilities
pub struct DeviceSelector;

impl DeviceSelector {
    #[cfg(feature = "metal")]
    fn try_metal_device() -> Result<Device, String> {
        match std::panic::catch_unwind(|| Device::new_metal(0)) {
            Ok(Ok(device)) => Ok(device),
            Ok(Err(err)) => Err(err.to_string()),
            Err(_) => Err("Metal device creation panicked".to_string()),
        }
    }

    #[cfg(feature = "cuda")]
    fn try_cuda_device() -> Result<Device, String> {
        match std::panic::catch_unwind(|| Device::new_cuda(0)) {
            Ok(Ok(device)) => Ok(device),
            Ok(Err(err)) => Err(err.to_string()),
            Err(_) => Err("CUDA device creation panicked".to_string()),
        }
    }

    /// Select the best available device based on preference
    pub fn select(preference: DevicePreference) -> candle_core::Result<Device> {
        match preference {
            DevicePreference::Cpu => {
                info!("Using CPU device (forced)");
                Ok(Device::Cpu)
            }
            DevicePreference::Gpu => Self::select_gpu(),
            DevicePreference::Auto => Self::select_auto(),
        }
    }

    /// Get or create a cached device using Auto preference
    pub fn get_cached_device() -> candle_core::Result<&'static Device> {
        DEVICE_CACHE.get_or_init(|| Self::select_auto().unwrap_or(Device::Cpu));
        Ok(DEVICE_CACHE.get().unwrap())
    }

    /// Force selection of a GPU device
    fn select_gpu() -> candle_core::Result<Device> {
        #[cfg(feature = "metal")]
        {
            match Self::try_metal_device() {
                Ok(device) => {
                    info!("Using Metal GPU device");
                    return Ok(device);
                }
                Err(e) => {
                    warn!("Failed to create Metal device: {}", e);
                }
            }
        }

        #[cfg(feature = "cuda")]
        {
            match Self::try_cuda_device() {
                Ok(device) => {
                    info!("Using CUDA GPU device");
                    return Ok(device);
                }
                Err(e) => {
                    warn!("Failed to create CUDA device: {}", e);
                }
            }
        }

        Err(candle_core::Error::msg(
            "No GPU device available or GPU features not enabled",
        ))
    }

    /// Automatically select the best available device
    fn select_auto() -> candle_core::Result<Device> {
        // Try GPU first if available
        match Self::select_gpu() {
            Ok(device) => Ok(device),
            Err(_) => {
                info!("GPU not available, falling back to CPU");
                Ok(Device::Cpu)
            }
        }
    }

    /// Check if a specific device type is available
    pub fn is_available(device_type: DeviceType) -> bool {
        match device_type {
            DeviceType::Cpu => true, // CPU is always available
            DeviceType::Metal => {
                #[cfg(feature = "metal")]
                {
                    Self::try_metal_device().is_ok()
                }
                #[cfg(not(feature = "metal"))]
                {
                    false
                }
            }
            DeviceType::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    Self::try_cuda_device().is_ok()
                }
                #[cfg(not(feature = "cuda"))]
                {
                    false
                }
            }
        }
    }

    /// Get information about available devices
    pub fn device_info() -> DeviceInfo {
        DeviceInfo {
            cpu_available: true,
            metal_available: Self::is_available(DeviceType::Metal),
            cuda_available: Self::is_available(DeviceType::Cuda),
        }
    }
}

/// Device types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    Cpu,
    Metal,
    Cuda,
}

/// Information about available devices
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub cpu_available: bool,
    pub metal_available: bool,
    pub cuda_available: bool,
}

impl DeviceInfo {
    /// Get the best available device type
    pub fn best_device_type(&self) -> DeviceType {
        if self.metal_available {
            DeviceType::Metal
        } else if self.cuda_available {
            DeviceType::Cuda
        } else {
            DeviceType::Cpu
        }
    }

    /// Check if any GPU is available
    pub fn has_gpu(&self) -> bool {
        self.metal_available || self.cuda_available
    }
}

impl std::fmt::Display for DeviceInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DeviceInfo {{ CPU: {}, Metal: {}, CUDA: {} }}",
            self.cpu_available, self.metal_available, self.cuda_available
        )
    }
}

/// Convenience function to create a device with Auto preference
pub fn auto_device() -> candle_core::Result<Device> {
    DeviceSelector::select(DevicePreference::Auto)
}

/// Convenience function to get a cached device
pub fn cached_device() -> candle_core::Result<&'static Device> {
    DeviceSelector::get_cached_device()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_preference_default() {
        assert_eq!(DevicePreference::default(), DevicePreference::Auto);
    }

    #[test]
    fn test_cpu_device_selection() {
        let device = DeviceSelector::select(DevicePreference::Cpu).unwrap();
        assert!(matches!(device, Device::Cpu));
    }

    #[test]
    fn test_device_info() {
        let info = DeviceSelector::device_info();
        assert!(info.cpu_available); // CPU should always be available

        // Test display implementation
        let display_str = format!("{}", info);
        assert!(display_str.contains("CPU: true"));
    }

    #[test]
    fn test_device_type_availability() {
        assert!(DeviceSelector::is_available(DeviceType::Cpu));
        // Metal/CUDA availability depends on features and platform
    }

    #[test]
    fn test_auto_device_function() {
        let device = auto_device();
        assert!(device.is_ok());
    }

    #[test]
    fn test_cached_device_function() {
        let device1 = cached_device();
        let device2 = cached_device();

        assert!(device1.is_ok());
        assert!(device2.is_ok());
        // Both should return the same device (pointer equality)
        assert_eq!(device1.unwrap() as *const _, device2.unwrap() as *const _);
    }
}
