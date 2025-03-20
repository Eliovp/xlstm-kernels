#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import triton
import math
from typing import Dict, Union, Any, Optional

from ..utils.kernels import is_power_of_2

# Import AMD detection and optimizations
try:
    from mlstm_kernels.triton.amd_detection import is_amd_gpu, is_mi300x, enable_amd_optimizations
    from mlstm_kernels.triton.amd_batch_aware import get_optimal_kernel_config, should_use_amd_optimized_kernel
    is_amd = is_amd_gpu()
    is_amd_mi300x = is_mi300x()
    if is_amd:
        enable_amd_optimizations()
except ImportError:
    is_amd = False
    is_amd_mi300x = False
    
    def get_optimal_kernel_config(*args, **kwargs):
        return {}
    
    def should_use_amd_optimized_kernel(*args, **kwargs):
        return False

# Flag for tests to override AMD detection
_is_amd_override = None

def is_amd_override(value: Optional[bool] = None) -> bool:
    """Get or set the AMD override flag."""
    global _is_amd_override
    if value is not None:
        _is_amd_override = value
    return _is_amd_override

def is_amd_hardware() -> bool:
    """
    Check if we're running on AMD hardware, with override option for testing.
    
    Returns:
        bool: True if AMD hardware is detected, False otherwise
    """
    if _is_amd_override is not None:
        return _is_amd_override
    return is_amd

def determine_kernel_heuristics(
    batch_size: int,
    seq_len: int,
    head_dim: int,
    forced_kernel: Optional[str] = None
) -> Dict[str, Union[int, str]]:
    """
    Determine kernel parameters based on input dimensions and hardware.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        head_dim: Head dimension
        forced_kernel: Override kernel selection
        
    Returns:
        Dict containing kernel parameters
    """
    # Start with default parameters
    params = {
        "BLOCK_SIZE_M": 32,
        "BLOCK_SIZE_N": 32, 
        "BLOCK_SIZE_K": 32,
        "USE_BATCH_STRIDED": batch_size > 1
    }
    
    # Check if we're running on AMD hardware
    if is_amd_hardware():
        # For AMD MI300X, use our optimized parameters
        if is_amd_mi300x:
            # Apply batch-aware optimizations
            if should_use_amd_optimized_kernel(batch_size, seq_len, head_dim):
                # Optimize for specific configurations
                if head_dim == 64:
                    # Head dimension 64 has special optimizations
                    params.update({
                        "BLOCK_SIZE_M": 64,  # Increased for better parallelism
                        "BLOCK_SIZE_N": 64,  # Increased for better parallelism
                        "BLOCK_SIZE_K": 32,  # Keep this dimension the same
                    })
                elif head_dim == 128:
                    # Head dimension 128 has different optimizations
                    params.update({
                        "BLOCK_SIZE_M": 64,
                        "BLOCK_SIZE_N": 128,
                        "BLOCK_SIZE_K": 32,
                    })
                else:
                    # For other head dimensions, use general AMD optimizations
                    params.update({
                        "BLOCK_SIZE_M": 32,
                        "BLOCK_SIZE_N": 64,
                        "BLOCK_SIZE_K": 32,
                    })
                
                # Add AMD-specific flags
                params["AMD_OPTIMIZED"] = True
                
                # Adjust based on the specific batch size and sequence length
                if batch_size >= 4:
                    params["GROUP_SIZE_M"] = 8  # Improve parallelism for large batches
                elif seq_len >= 4096:
                    params["GROUP_SIZE_M"] = 4  # Balance for long sequences
            else:
                # For configurations where native kernels perform better
                params["AMD_OPTIMIZED"] = False
        else:
            # For other AMD GPUs, use more conservative settings
            params["AMD_OPTIMIZED"] = False
    
    # Override with forced kernel if provided
    if forced_kernel:
        params["FORCED_KERNEL"] = forced_kernel
    
    return params

def get_optimized_kernel_config(
    batch_size: int, 
    seq_len: int,
    head_dim: int
) -> Dict[str, str]:
    """
    Get the optimal kernel configuration based on input dimensions and hardware.
    
    This is a convenience wrapper around our AMD-specific optimizations.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        head_dim: Head dimension
        
    Returns:
        Dict containing kernel configuration
    """
    # Check if we're on AMD hardware
    if is_amd_hardware() and is_amd_mi300x:
        # Get optimal kernel configuration from our AMD module
        return get_optimal_kernel_config(batch_size, seq_len, head_dim)
    
    # Default to native kernels for non-AMD hardware
    return {
        "chunkwise_kernel": "chunkwise--native_autograd",
        "sequence_kernel": "native_sequence__native",
        "step_kernel": "native"
    }
