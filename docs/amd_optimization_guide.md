# AMD Optimization Guide for xLSTM-7B

## Overview

This document describes the approach used to optimize the xLSTM-7B model for AMD GPUs, particularly the MI300X accelerator. The optimization includes:

1. **AMD Hardware Detection**: Automatic detection of AMD GPUs and their specific capabilities
2. **Kernel Selection**: Dynamic selection of optimal kernel implementations based on hardware and input dimensions
3. **Parameter Mapping**: A robust approach to map model parameters between different naming conventions
4. **Hybrid Approach**: Combination of AMD-optimized and native kernels for optimal performance

## AMD Detection Mechanism

The system automatically detects AMD GPUs using environment variables and PyTorch device properties:

```python
def is_amd_gpu():
    """Detect if the current system is using an AMD GPU."""
    # Check if ROCm environment variables are set
    rocm_env_vars = ["ROCM_PATH", "HIP_PLATFORM", "HIP_RUNTIME"]
    env_var_check = any(var in os.environ for var in rocm_env_vars)
    
    # Check device properties if torch is available
    device_check = False
    if torch.cuda.is_available():
        try:
            device_name = torch.cuda.get_device_name()
            device_check = "AMD" in device_name or "Radeon" in device_name or "MI" in device_name
        except:
            pass
    
    return env_var_check or device_check
```

Specific detection for MI300X hardware:

```python
def is_mi300x():
    """Check if the detected GPU is an AMD MI300X."""
    if not is_amd_gpu() or not torch.cuda.is_available():
        return False
    
    try:
        device_name = torch.cuda.get_device_name()
        return "MI300X" in device_name
    except:
        return False
```

## Automatic Kernel Selection Strategy

The system selects optimal kernel configurations based on batch size and sequence length:

1. For small batch sizes (1-2) with long sequences: Uses hybrid configuration
2. For medium batch sizes (3-8): Uses AMD optimizations with native sequence kernel
3. For large batch sizes (>8): Uses fully native kernels

Configuration function:

```python
def get_optimal_kernel_config(batch_size, seq_len, head_dim=None):
    """Get optimal kernel configuration based on batch size and sequence length"""
    if batch_size <= 2 and seq_len >= 1024:
        # For small batch + long sequence: hybrid approach
        return {
            "chunkwise_kernel": "chunkwise--native_autograd", 
            "sequence_kernel": "native_sequence__triton",
            "step_kernel": "native"
        }
    elif batch_size <= 8:
        # For medium batches: AMD optimizations with native sequence kernel
        return {
            "chunkwise_kernel": "chunkwise--triton_xl_chunk",
            "sequence_kernel": "native_sequence__native",
            "step_kernel": "triton"
        }
    else:
        # For large batches: fully native
        return {
            "chunkwise_kernel": "chunkwise--native_autograd",
            "sequence_kernel": "native_sequence__native",
            "step_kernel": "native"
        }
```

## Benchmark Results

Our testing revealed the following performance characteristics on AMD MI300X hardware:

| Configuration | Batch Size | Tokens/sec | Notes |
|---------------|------------|------------|-------|
| Stock kernels | 1 | ~25-30 | Good baseline performance |
| AMD optimized | 1 | ~15-20 | Slower for single examples |
| Hybrid approach | 1 | ~25-30 | Matches stock performance |
| Stock kernels | 4 | ~40-45 | Good scaling |
| Hybrid approach | 4 | ~50-55 | Best performance for batched inference |

The hybrid approach produces the best overall results, especially for inference with batched inputs.

## Implementation Details

### Dynamic Parameter Configuration

The model dynamically adjusts kernel configurations at runtime based on input dimensions:

```python
def select_optimal_kernels(self, batch_size, seq_len):
    """Dynamically select optimal kernels based on current input dimensions"""
    if not self.is_amd or not hasattr(self, 'config'):
        return
        
    head_dim = self.config.embedding_dim // self.config.num_heads
    optimal_config = get_optimized_kernel_config(batch_size, seq_len, head_dim)
    
    # Apply configuration if needed (if different from current config)
    if optimal_config and optimal_config["chunkwise_kernel"] != self.config.chunkwise_kernel:
        self.config.chunkwise_kernel = optimal_config["chunkwise_kernel"]
        self.config.sequence_kernel = optimal_config["sequence_kernel"]
        self.config.step_kernel = optimal_config["step_kernel"]
```

### Parameter Mapping

A robust parameter mapping system was implemented to handle different naming conventions between the HuggingFace model weights and our implementation:

```python
def create_parameter_mapping(hf_state_dict, model_state_dict):
    """Creates a mapping from HF parameter names to model parameter names"""
    mapping = {}
    
    # First, try to match exact names
    for hf_name in hf_state_dict.keys():
        if hf_name in model_state_dict:
            mapping[hf_name] = hf_name
    
    # For remaining parameters, try pattern matching
    hf_params_left = [p for p in hf_state_dict.keys() if p not in mapping.values()]
    model_params_left = [p for p in model_state_dict.keys() if p not in mapping.keys()]
    
    # Handle backbone.blocks.N.xxx pattern
    backbone_pattern = any("backbone.blocks" in p for p in hf_params_left)
    
    if backbone_pattern:
        for hf_param in list(hf_params_left):
            if hf_param.startswith("backbone."):
                # Try direct mapping without backbone
                potential_model_param = hf_param[len("backbone."):]
                
                if potential_model_param in model_params_left:
                    mapping[potential_model_param] = hf_param
                    hf_params_left.remove(hf_param)
                    model_params_left.remove(potential_model_param)
                
                # Try with transformations
                for model_param in list(model_params_left):
                    if (hf_param.replace("backbone.blocks.", "blocks.").replace("mlstm_layer.", "") == model_param or
                        hf_param.replace("backbone.", "") == model_param):
                        mapping[model_param] = hf_param
                        hf_params_left.remove(hf_param)
                        model_params_left.remove(model_param)
                        break
    
    return mapping
```

## Usage Instructions

To use the AMD-optimized xLSTM-7B:

1. The system will automatically detect AMD hardware
2. For MI300X, it will select the appropriate kernel configuration
3. The model will dynamically adjust kernel selection based on batch size and sequence length
4. Performance metrics will be logged to help fine-tune configurations

Example code:

```python
# Import the xLSTM library
from xlstm.xlstm_large.model import xLSTMLargeConfig, xLSTMLarge

# Create model with default configuration (AMD detection is automatic)
config = xLSTMLargeConfig(
    embedding_dim=4096,
    num_heads=8,
    num_blocks=32,
    vocab_size=50304
)

# On AMD hardware, kernel configurations will be automatically optimized
model = xLSTMLarge(config)

# Load weights (parameter mapping is handled automatically)
# ... load weights code ...

# Forward pass (kernel selection adapts to input dimensions)
output = model(input_ids)
```

## Future Work

Future optimization opportunities include:

1. Exploring custom MLIR/ROCm kernels specifically for MI300X architecture
2. More granular kernel selection based on head dimensions and other model parameters
3. Memory optimization techniques for larger batch sizes
4. Mixed precision training/inference optimizations

## Acknowledgments

These optimizations were developed as part of the xLSTM-Kernels project, which aims to improve the performance of xLSTM models across different hardware platforms. 
