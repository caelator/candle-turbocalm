fn main() {
    let device = candle_core::Device::Cpu;
    let a = candle_core::Tensor::ones((2, 3, 4), candle_core::DType::F32, &device).unwrap();
    let b = candle_core::Tensor::ones((4, 4), candle_core::DType::F32, &device).unwrap();
    
    // In candle, to multiply [2,3,4] with [4,4], b might need to be broadcasted to [2,4,4].
    // Let's try to compile and run.
    match a.matmul(&b) {
        Ok(t) => println!("Success: {:?}", t.dims()),
        Err(e) => println!("Error: {}", e),
    }
}
