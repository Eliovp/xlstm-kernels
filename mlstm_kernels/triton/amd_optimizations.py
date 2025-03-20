#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

"""AMD-specific optimizations for mlstm kernels on CDNA3.0 (gfx942) architecture."""

import triton
import triton.language as tl
import torch
import os
from typing import Callable, Dict, Optional, Tuple, Union

def is_amd() -> bool:
    """Check if the current hardware is AMD with HIP backend."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0).lower()
        return "amd" in device_name or "mi" in device_name or "instinct" in device_name
    return False

def is_cdna3() -> bool:
    """Check if the current AMD GPU is CDNA3 architecture (MI300)."""
    if not is_amd():
        return False
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0).lower()
        return "mi300" in device_name or "mi 300" in device_name
    return False

def get_amd_warp_size() -> int:
    """Get the warp/wavefront size for AMD GPUs."""
    # AMD GPUs use a wavefront size of 64
    return 64

def get_amd_num_cu() -> int:
    """Get the number of Compute Units available on the AMD GPU."""
    if not is_amd():
        return 0
    
    # For MI300X, there are 228 CUs
    if is_cdna3():
        return 228
    
    # Default fallback for other AMD GPUs
    return 120

def get_amd_block_size(head_dim: int, min_block_size: int = 64, max_block_size: int = 256) -> int:
    """
    Determine the optimal block size for AMD GPUs.
    
    Args:
        head_dim: The head dimension size
        min_block_size: Minimum block size (default: 64 to match wavefront size)
        max_block_size: Maximum block size (default: 256 to avoid register pressure)
    
    Returns:
        An optimal block size for AMD GPUs that's a multiple of 64
    """
    # For CDNA3.0 architecture (MI300), optimize based on head dimensions
    if is_cdna3():
        if head_dim <= 64:
            # For small head dimensions (64), use 32 for better occupancy and efficiency
            # This helps avoid the slowdowns we observed with head_dim=64
            target_size = 32
        elif head_dim <= 128:
            # For medium head dimensions, use 128 (two wavefronts)
            target_size = 128
        else:
            # For large head dimensions, use 192 (three wavefronts)
            target_size = 192
    else:
        # For other AMD GPUs, use a tuned block size based on head dimension
        if head_dim <= 64:
            target_size = 32
        else:
            target_size = max(min_block_size, min(head_dim, max_block_size))
    
    # For head_dim=64, we know from benchmarks that smaller blocks work better
    # instead of aligning to wavefront size
    if head_dim == 64:
        return target_size
    
    # For other dimensions, ensure block size is a multiple of wavefront size (64)
    warp_size = get_amd_warp_size()
    target_size = ((target_size + warp_size - 1) // warp_size) * warp_size
    
    # Constrain to min/max range
    return max(min(target_size, max_block_size), 16)  # Allow smaller min size for head_dim=64

def get_amd_kernel_config(head_dim_q: int, head_dim_v: Optional[int] = None) -> Dict[str, int]:
    """
    Generate optimal kernel configurations for AMD GPUs.
    
    Args:
        head_dim_q: The query head dimension
        head_dim_v: The value head dimension (defaults to head_dim_q if not provided)
    
    Returns:
        A dictionary with optimal kernel configuration parameters
    """
    if head_dim_v is None:
        head_dim_v = head_dim_q
    
    warp_size = get_amd_warp_size()
    block_size_q = get_amd_block_size(head_dim_q)
    block_size_v = get_amd_block_size(head_dim_v)
    
    # For CDNA3.0 architecture (MI300), use specifically tuned parameters
    if is_cdna3():
        # Special handling for head_dim=64 which showed slower performance
        if head_dim_q == 64:
            return {
                "BLOCK_SIZE_Q": 32,  # Smaller block size for better performance
                "BLOCK_SIZE_V": 32,
                "BLOCK_SIZE_K": 32,   # Smaller K dimension blocks
                "STAGE_SIZE": 2,      # Smaller stages for reduced overhead
                "CHUNK_SIZE": 64      # Smaller chunk size for better cache utilization
            }
        elif head_dim_q <= 64:
            # For small head dimensions, optimize for memory coalescing
            return {
                "BLOCK_SIZE_Q": block_size_q,
                "BLOCK_SIZE_V": block_size_v,
                "BLOCK_SIZE_K": 32,  # Smaller for better occupancy
                "STAGE_SIZE": 2,     # Smaller stage size for better latency hiding
                "CHUNK_SIZE": 64     # Smaller chunks for head_dim <= 64
            }
        elif head_dim_q <= 128:
            # For medium head dimensions
            return {
                "BLOCK_SIZE_Q": block_size_q,
                "BLOCK_SIZE_V": block_size_v,
                "BLOCK_SIZE_K": warp_size,
                "STAGE_SIZE": 3,
                "CHUNK_SIZE": 128
            }
        else:
            # For large head dimensions, use more aggressive tiling
            return {
                "BLOCK_SIZE_Q": block_size_q,
                "BLOCK_SIZE_V": block_size_v,
                "BLOCK_SIZE_K": warp_size * 2,  # Larger blocks for K dimension
                "STAGE_SIZE": 2,  # Smaller stage for larger dimensions
                "CHUNK_SIZE": 128
            }
    
    # Default configuration for other AMD GPUs
    return {
        "BLOCK_SIZE_Q": block_size_q,
        "BLOCK_SIZE_V": block_size_v,
        "BLOCK_SIZE_K": warp_size,
        "STAGE_SIZE": 3,
        "CHUNK_SIZE": 128
    }

def optimize_grid_for_amd(grid_fn: Callable) -> Callable:
    """
    Wrapper to optimize grid calculation for AMD hardware.
    
    This ensures grid dimensions are aligned with wavefront size
    for better hardware utilization.
    
    Args:
        grid_fn: Original grid calculation function
        
    Returns:
        Optimized grid calculation function for AMD hardware
    """
    def optimized_grid_fn(*args, **kwargs):
        grid = grid_fn(*args, **kwargs)
        
        if is_amd():
            warp_size = get_amd_warp_size()
            
            # Ensure grid dimensions are multiples of wavefront size
            # for better occupancy on AMD hardware
            if isinstance(grid, tuple) and len(grid) >= 2:
                grid_x, grid_y = grid[0:2]
                
                # For head_dim=64 cases, we want more grid elements with smaller blocks
                # Don't round to warp_size for these cases as it reduces parallelism
                if "head_dim" in kwargs and kwargs["head_dim"] == 64:
                    # Increase grid size for better parallelism with smaller blocks
                    grid_x = grid_x * 2
                    
                    # Use original grid values without rounding to warp_size
                    if len(grid) > 2:
                        return (grid_x, grid_y, *grid[2:])
                    return (grid_x, grid_y)
                
                # For other cases, align grid to wavefront size
                grid_x = ((grid_x + warp_size - 1) // warp_size) * warp_size
                grid_y = ((grid_y + warp_size - 1) // warp_size) * warp_size
                
                if len(grid) > 2:
                    return (grid_x, grid_y, *grid[2:])
                return (grid_x, grid_y)
            
        return grid
    
    return optimized_grid_fn

def get_amd_transpose_chunk_size(seq_len: int, head_dim: int) -> int:
    """
    Determine the optimal chunk size for transpose operations on AMD GPUs.
    
    Args:
        seq_len: Sequence length
        head_dim: Head dimension
        
    Returns:
        Optimal chunk size for AMD transpose operations
    """
    warp_size = get_amd_warp_size()
    
    # Specialized for head_dim=64
    if head_dim == 64:
        if seq_len <= 2048:
            return 32  # Smaller chunks for better performance
        elif seq_len <= 8192:
            return 64
        else:
            return 128
    
    # For MI300, optimize based on sequence length and head dimension
    if is_cdna3():
        if seq_len <= 2048:
            if head_dim <= 64:
                return 64
            elif head_dim <= 128:
                return 128
            else:
                return 256
        elif seq_len <= 8192:
            if head_dim <= 64:
                return 128
            else:
                return 256
        else:
            return 512
    
    # Default for other AMD GPUs
    return max(64, ((head_dim + warp_size - 1) // warp_size) * warp_size)

def tune_amd_memory_access(head_dim: int) -> Dict[str, Union[bool, int]]:
    """
    Provide tuning parameters for memory access patterns on AMD GPUs.
    
    Args:
        head_dim: Head dimension
        
    Returns:
        Dictionary of memory access optimization parameters
    """
    # Special case for head_dim=64 based on benchmark results
    if head_dim == 64:
        return {
            "vectorize_load": False,     # Avoid vectorization for head_dim=64
            "vectorize_width": 1,        # No vectorization
            "use_shared_memory": True,   # Still use shared memory
            "prefetch_factor": 1,        # Minimum prefetch to reduce overhead
            "transpose_shared_memory": True,  # Use shared memory for transposes
            "reduce_bank_conflicts": True      # Avoid bank conflicts
        }
    
    # For CDNA3 architecture (MI300)
    if is_cdna3():
        if head_dim < 64:
            return {
                "vectorize_load": False,      # Avoid vectorization for very small dimensions
                "vectorize_width": 1,
                "use_shared_memory": True,
                "prefetch_factor": 1
            }
        elif head_dim <= 128:
            return {
                "vectorize_load": True,
                "vectorize_width": 2,  # Use float2 loads
                "use_shared_memory": True,
                "prefetch_factor": 2
            }
        else:
            return {
                "vectorize_load": False,  # Avoid vectorization for large dimensions
                "vectorize_width": 1,
                "use_shared_memory": True,
                "prefetch_factor": 1
            }
    
    # Default for other AMD GPUs
    return {
        "vectorize_load": head_dim <= 128 and head_dim != 64,  # Skip for head_dim=64
        "vectorize_width": 2 if head_dim <= 128 and head_dim != 64 else 1,
        "use_shared_memory": True,
        "prefetch_factor": 1
    } 