# AMD Optimization Guide for xLSTM

This guide documents the optimizations implemented for xLSTM kernels on AMD MI300X hardware.

## Overview

The xLSTM library now includes automatic detection and optimization for AMD GPUs, particularly the MI300X model. The system will:

1. Automatically detect AMD GPUs at runtime
2. Apply appropriate optimizations based on the detected hardware
3. Dynamically adjust kernel selection based on batch size and sequence length
4. Fall back to native kernels when they perform better than Triton kernels

## AMD Detection

The system detects AMD hardware through multiple mechanisms:

- Environment variables (ROCM_PATH, HIP_PATH)
- PyTorch build information (torch.version.hip)
- Device properties inspection
- Version string checking

The detection logic is in `mlstm_kernels/triton/amd_detection.py`.

## Automatic Kernel Selection

Based on our benchmarks, we've implemented a batch-aware kernel selection strategy that chooses the optimal kernel configuration for different combinations of:

- Batch size
- Sequence length
- Head dimension

This logic is implemented in `mlstm_kernels/triton/amd_batch_aware.py`.

### Benchmark Results

For models with head dimension 64 and 128, we found the following optimal configurations:

| Batch Size | Sequence Length | Best Kernel Type | Speedup |
|------------|-----------------|------------------|---------|
| 1          | 2048            | Native           | -       |
| 1          | 4096            | Native           | -       |
| 2          | 2048            | AMD-optimized    | 1.17x   |
| 2          | 4096            | Native           | -       |
| 4          | 2048            | AMD-optimized    | 1.11x   |
| 4          | 4096            | AMD-optimized    | 1.08x   |

Interestingly, we found that AMD-optimized kernels yield better performance in specific configurations, while native kernels perform better in others.

## Implementation

The implementation consists of the following components:

### 1. AMD Detection Module

```python
# mlstm_kernels/triton/amd_detection.py
def is_amd_gpu():
    """Detect if the current system is using an AMD GPU."""
    # ...

def is_mi300x():
    """Specifically detect if the GPU is an AMD MI300X."""
    # ...

def enable_amd_optimizations():
    """Configure PyTorch to use optimal settings for AMD GPUs."""
    # ...
```

### 2. Batch-Aware Kernel Selection

```python
# mlstm_kernels/triton/amd_batch_aware.py
def get_optimal_kernel_config(batch_size, seq_len, head_dim=None):
    """Returns the optimal kernel configuration based on input dimensions."""
    # ...

def should_use_amd_optimized_kernel(batch_size, seq_len, head_dim=None):
    """Determines if AMD-optimized kernels should be used."""
    # ...
```

### 3. Integration with xLSTM

The xLSTM model has been updated to:

1. Detect AMD hardware during initialization
2. Apply appropriate initial kernel configuration
3. Dynamically adjust kernel selection during forward passes based on input dimensions

```python
# xlstm/xlstm_large/model.py
def forward(self, x):
    # For AMD hardware, dynamically select optimal kernels
    if self.is_amd and self.is_amd_mi300x:
        batch_size = x.size(0)
        seq_len = x.size(1)
        self.select_optimal_kernels(batch_size, seq_len)
    
    # Continue with normal forward pass
    # ...
```

## Usage

When using xLSTM on AMD hardware, no changes to your code are needed. The library will automatically detect the hardware and apply optimizations.

If you want to force a specific kernel configuration, you can do so by setting the appropriate parameters in the xLSTMLargeConfig:

```python
from xlstm.xlstm_large.model import xLSTMLargeConfig, xLSTMLarge

# Use specific kernel configuration
config = xLSTMLargeConfig(
    embedding_dim=4096,
    num_heads=32,
    num_blocks=32,
    # Force specific kernel types
    chunkwise_kernel="chunkwise--native_autograd",
    sequence_kernel="native_sequence__native",
    step_kernel="native"
)

model = xLSTMLarge(config)
```

## Testing

The repository includes several test scripts to validate the AMD optimizations:

- `test_xlstm_hybrid.py` - Tests different kernel configurations side by side
- `test_xlstm_auto.py` - Tests the automatic kernel selection system
- `test_xlstm_hybrid_full.py` - Tests the full xLSTM-7B model with dynamic kernel selection

## Performance Results

Performance varies significantly based on configuration. Our testing shows that for the xLSTM-7B model:

1. **Small batch sizes (B=1)**: Native kernels generally perform better
2. **Medium batch sizes (B=2)**: AMD-optimized kernels show up to 1.17x speedup
3. **Large batch sizes (B=4)**: AMD-optimized kernels show up to 1.11x speedup, particularly for sequence length 2048
4. **Long sequences (S=4096)**: AMD-optimized kernels perform better at larger batch sizes (B=4)

## Future Work

Opportunities for additional optimizations:

1. Expand the optimization to cover more configurations
2. Implement AMD-specific BLAS optimizations for matrix multiplications
3. Further tune GEMM configurations for head dimensions other than 64 and 128
4. Implement advanced memory access patterns optimized for AMD's cache hierarchy
5. Explore ROCm-specific optimizations for the underlying PyTorch operations 