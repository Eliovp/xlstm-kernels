"""
AMD-specific optimizations for different batch sizes and sequence lengths
Based on our benchmark results.
"""

import logging

logger = logging.getLogger(__name__)

def get_optimal_kernel_config(batch_size, seq_len, head_dim=None):
    """
    Returns the optimal kernel configuration based on batch size and sequence length
    for AMD MI300X GPUs.
    
    Args:
        batch_size: int, batch size of the input
        seq_len: int, sequence length of the input
        head_dim: int, optional, head dimension if known
        
    Returns:
        dict: Configuration parameters for optimal performance
    """
    # For head dimension 64, we have specific optimizations
    if head_dim == 64 or head_dim == 128:
        # Based on our benchmark results, optimized only for specific configurations
        if batch_size == 2 and seq_len == 2048:
            logger.info(f"Using AMD-optimized kernel for B={batch_size}, S={seq_len}, H={head_dim}")
            return {
                "chunkwise_kernel": "chunkwise--triton_xl_chunk",
                "sequence_kernel": "native_sequence__triton",
                "step_kernel": "triton"
            }
        elif batch_size == 4 and seq_len == 2048:
            logger.info(f"Using AMD-optimized kernel for B={batch_size}, S={seq_len}, H={head_dim}")
            return {
                "chunkwise_kernel": "chunkwise--triton_xl_chunk",
                "sequence_kernel": "native_sequence__triton",
                "step_kernel": "triton"
            }
        elif batch_size == 4 and seq_len == 4096:
            logger.info(f"Using AMD-optimized kernel for B={batch_size}, S={seq_len}, H={head_dim}")
            return {
                "chunkwise_kernel": "chunkwise--triton_xl_chunk",
                "sequence_kernel": "native_sequence__triton",
                "step_kernel": "triton"
            }
        
    # For all other cases, our benchmarks show native kernels perform better
    logger.info(f"Using native kernels for B={batch_size}, S={seq_len}, H={head_dim if head_dim else 'unknown'}")
    return {
        "chunkwise_kernel": "chunkwise--native_autograd",
        "sequence_kernel": "native_sequence__native",
        "step_kernel": "native"
    }

def get_amd_grid_config(batch_size, seq_len, head_dim=None):
    """
    Returns the optimal grid configuration for AMD MI300X GPUs.
    
    Args:
        batch_size: int, batch size of the input
        seq_len: int, sequence length of the input
        head_dim: int, optional, head dimension if known
        
    Returns:
        dict: Grid configuration parameters
    """
    # Different block sizes based on batch size and sequence length
    if batch_size >= 4:
        # For larger batch sizes, use larger blocks
        return {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32
        }
    elif seq_len >= 4096:
        # For longer sequences, optimize differently
        return {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32
        }
    else:
        # Default configuration
        return {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32
        }

def should_use_amd_optimized_kernel(batch_size, seq_len, head_dim=None):
    """
    Determines if AMD-optimized kernels should be used based on our benchmark results.
    
    Args:
        batch_size: int, batch size of the input
        seq_len: int, sequence length of the input
        head_dim: int, optional, head dimension if known
        
    Returns:
        bool: True if AMD-optimized kernels should be used, False otherwise
    """
    # Based on our comprehensive benchmark results
    if head_dim == 64 or head_dim == 128:
        # Batch 2, seq 2048 - showed 1.17x speedup
        if batch_size == 2 and seq_len == 2048:
            return True
            
        # Batch 4, seq 2048 - showed 1.11x speedup in some tests
        elif batch_size == 4 and seq_len == 2048:
            return True
            
        # Batch 4, seq 4096 - showed 1.08x speedup
        elif batch_size == 4 and seq_len == 4096:
            return True
    
    # For all other configurations, our tests showed that native kernels 
    # performed better or approximately the same
    return False 