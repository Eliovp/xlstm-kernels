#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import triton

from ..utils.kernels import is_power_of_2

# Import AMD-specific optimizations if available
try:
    from .amd_optimizations import is_amd as _original_is_amd
    from .amd_optimizations import get_amd_block_size

    # Allow for testing with/without AMD optimizations
    is_amd_override = None
    
    def is_amd():
        if is_amd_override is not None:
            return is_amd_override()
        return _original_is_amd()
        
    AMD_SUPPORT = True
except ImportError:
    AMD_SUPPORT = False
    
    def is_amd():
        return False
    
    def get_amd_block_size(head_dim, min_block_size=64):
        return min_block_size

# Import this to allow the function to be patched for comparison testing
import sys
current_module = sys.modules[__name__]

def get_head_dim_block_size(head_dim: int, min_block_size: int = 64) -> int:
    # TODO make proper tests, for when and where this check is necessary.
    # For 160M model size, this check is not necessary.
    # assert (
    #     is_power_of_2(head_dim) or head_dim % min_block_size == 0
    # ), f"head_dim must be a power of 2 or multiple of {min_block_size}. Got {head_dim}."
    
    # Check for AMD GPU and use AMD-specific block sizes
    if hasattr(current_module, "is_amd_override"):
        is_amd_result = current_module.is_amd_override()
    else:
        is_amd_result = is_amd()
        
    if is_amd_result:
        return get_amd_block_size(head_dim)
    
    # Original NVIDIA-optimized code
    if head_dim <= 64:
        return 64
    elif head_dim <= 128:
        return 128
    else:
        return head_dim
