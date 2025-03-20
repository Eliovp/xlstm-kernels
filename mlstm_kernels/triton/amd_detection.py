"""
Utility module for AMD hardware detection and configuration.
"""

import torch
import os
import logging
import platform

logger = logging.getLogger(__name__)

def is_amd_gpu():
    """
    Detect if the current system is using an AMD GPU.
    
    Returns:
        bool: True if AMD GPU is detected, False otherwise.
    """
    # Check environment variable that might indicate AMD
    if os.environ.get("ROCM_PATH") or os.environ.get("HIP_PATH"):
        logger.info("Detected AMD GPU via environment variables")
        return True
    
    # Check if ROCm/HIP is in the PyTorch build
    if torch.version.hip is not None:
        logger.info("Detected AMD GPU via PyTorch ROCm/HIP build")
        return True
    
    # Try to detect through device properties if CUDA is available
    if torch.cuda.is_available():
        try:
            for i in range(torch.cuda.device_count()):
                device_props = torch.cuda.get_device_properties(i)
                # AMD GPUs in ROCm environment often have specific identifiers
                if "AMD" in device_props.name or "Radeon" in device_props.name or "MI" in device_props.name:
                    logger.info(f"Detected AMD GPU: {device_props.name}")
                    return True
        except Exception as e:
            logger.warning(f"Error checking GPU properties: {e}")
    
    # Additional check for AMD GPUs via platform info
    try:
        # This is a fallback and may not be reliable
        if "rocm" in torch.__version__.lower():
            logger.info("Detected AMD GPU via PyTorch version string")
            return True
    except:
        pass
    
    return False

def get_amd_info():
    """
    Get detailed information about AMD GPU if present.
    
    Returns:
        dict: Information about the AMD GPU or None if not detected
    """
    if not is_amd_gpu():
        return None
    
    gpu_info = {}
    
    # Try to get more detailed GPU info
    if torch.cuda.is_available():
        try:
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpu_info[f"gpu_{i}"] = {
                    "name": props.name,
                    "total_memory": props.total_memory,
                    "multi_processor_count": props.multi_processor_count,
                    "compute_capability": f"{props.major}.{props.minor}"
                }
        except Exception as e:
            logger.warning(f"Error obtaining detailed AMD GPU info: {e}")
    
    # Add environment information
    gpu_info["environment"] = {
        "rocm_path": os.environ.get("ROCM_PATH", "Not set"),
        "hip_path": os.environ.get("HIP_PATH", "Not set"),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__
    }
    
    return gpu_info

def is_mi300x():
    """
    Specifically detect if the GPU is an AMD MI300X.
    
    Returns:
        bool: True if MI300X is detected, False otherwise
    """
    if not is_amd_gpu():
        return False
    
    # Check for MI300X through device name
    if torch.cuda.is_available():
        try:
            for i in range(torch.cuda.device_count()):
                device_props = torch.cuda.get_device_properties(i)
                if "MI300" in device_props.name or "MI300X" in device_props.name:
                    logger.info(f"Detected AMD MI300X: {device_props.name}")
                    return True
        except Exception as e:
            logger.warning(f"Error checking for MI300X: {e}")
    
    # Check environment variable that might be explicitly set
    if os.environ.get("AMD_MI300X") == "1":
        logger.info("Detected AMD MI300X via environment variable")
        return True
    
    return False

def enable_amd_optimizations():
    """
    Configure PyTorch to use optimal settings for AMD GPUs.
    
    Returns:
        bool: True if optimizations were applied, False otherwise
    """
    if not is_amd_gpu():
        return False
    
    # Apply AMD-specific optimizations
    try:
        # Set environment variables that might affect AMD performance
        os.environ["MIOPEN_DEBUG_CONV_IMPLICIT_GEMM"] = "0"  # Use implicit GEMM
        os.environ["MIOPEN_FIND_MODE"] = "1"  # Use cached kernels
        
        if torch.cuda.is_available():
            # Enable TF32 if available (similar to NVIDIA)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # Benchmark to find optimal algorithms
            torch.backends.cudnn.benchmark = True
            
        logger.info("Applied AMD GPU optimizations")
        return True
    except Exception as e:
        logger.warning(f"Error applying AMD optimizations: {e}")
        return False 