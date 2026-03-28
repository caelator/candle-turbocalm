use candle_core::{Result, Tensor, Device};

pub fn pack_bits(tensor: &Tensor, bits: u8) -> Result<Tensor> {
    let flattened = tensor.flatten_all()?.to_vec1::<u8>()?;
    let mut packed = Vec::new();
    
    if bits == 4 {
        for chunk in flattened.chunks(2) {
            let high = chunk[0] & 0x0F;
            let low = if chunk.len() > 1 { chunk[1] & 0x0F } else { 0 };
            packed.push((high << 4) | low);
        }
    } else if bits == 2 {
        for chunk in flattened.chunks(4) {
            let mut byte = 0;
            for (i, &val) in chunk.iter().enumerate() {
                byte |= (val & 0x03) << (6 - i * 2);
            }
            packed.push(byte);
        }
    } else if bits == 1 {
        for chunk in flattened.chunks(8) {
            let mut byte = 0;
            for (i, &val) in chunk.iter().enumerate() {
                byte |= (val & 0x01) << (7 - i);
            }
            packed.push(byte);
        }
    } else {
        packed = flattened;
    }
    
    let packed_len = packed.len();
    Tensor::from_vec(packed, (packed_len,), tensor.device())
}

pub fn unpack_bits(tensor: &Tensor, bits: u8, original_shape: &[usize]) -> Result<Tensor> {
    let packed = tensor.to_vec1::<u8>()?;
    let mut unpacked = Vec::new();
    
    let target_elements = original_shape.iter().product::<usize>();
    
    if bits == 4 {
        for &byte in &packed {
            unpacked.push((byte >> 4) & 0x0F);
            unpacked.push(byte & 0x0F);
        }
    } else if bits == 2 {
        for &byte in &packed {
            unpacked.push((byte >> 6) & 0x03);
            unpacked.push((byte >> 4) & 0x03);
            unpacked.push((byte >> 2) & 0x03);
            unpacked.push(byte & 0x03);
        }
    } else if bits == 1 {
        for &byte in &packed {
            for i in 0..8 {
                unpacked.push((byte >> (7 - i)) & 0x01);
            }
        }
    } else {
        unpacked = packed;
    }
    
    unpacked.truncate(target_elements);
    Tensor::from_vec(unpacked, original_shape, tensor.device())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pack_unpack_4bit() -> Result<()> {
        let device = Device::Cpu;
        let data: Vec<u8> = vec![5, 12, 0, 15, 3, 8];
        let tensor = Tensor::from_vec(data.clone(), (2, 3), &device)?;
        
        let packed = pack_bits(&tensor, 4)?;
        assert_eq!(packed.dims(), &[3]);
        
        let unpacked = unpack_bits(&packed, 4, &[2, 3])?;
        let unpacked_vec = unpacked.to_vec2::<u8>()?;
        
        assert_eq!(unpacked_vec[0], vec![5, 12, 0]);
        assert_eq!(unpacked_vec[1], vec![15, 3, 8]);
        
        Ok(())
    }
    
    #[test]
    fn test_pack_unpack_1bit() -> Result<()> {
        let device = Device::Cpu;
        let data: Vec<u8> = vec![1, 0, 1, 1, 0, 0, 1, 0, 1, 1];
        let tensor = Tensor::from_vec(data.clone(), (10,), &device)?;
        
        let packed = pack_bits(&tensor, 1)?;
        assert_eq!(packed.dims(), &[2]);
        
        let unpacked = unpack_bits(&packed, 1, &[10])?;
        let unpacked_vec = unpacked.to_vec1::<u8>()?;
        
        assert_eq!(unpacked_vec, data);
        
        Ok(())
    }
}
