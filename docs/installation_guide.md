# Installation and Usage Guide for AMD-Optimized xLSTM

This guide provides detailed instructions for installing and using the AMD-optimized xLSTM library with automatic hardware detection.

## Installation

### Option 1: Clone the Repository (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/xlstm-kernels.git
cd xlstm-kernels

# Install dependencies
pip install -e .

# For benchmark visualization dependencies
pip install matplotlib tabulate
```

### Option 2: Install from Your Branch

If you have already cloned the repository and want to switch to the AMD optimization branch:

```bash
# Navigate to your existing repository
cd xlstm-kernels

# Switch to the AMD optimization branch
git fetch
git checkout feature/amd_optimizations

# Install dependencies
pip install -e .
```

## Basic Usage

The library automatically detects AMD hardware and applies appropriate optimizations. You don't need to make any changes to your code - it just works!

```python
# Import the library
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

# Load weights
# (weights loading code...)

# Forward pass (kernel selection adapts to input dimensions automatically)
output = model(input_ids)
```

## Running Benchmarks

To compare performance across different kernel configurations, use the benchmark script:

```bash
# Run benchmarks with small model (quick testing)
python benchmark_comparison.py --small-model

# Run benchmarks with full model (longer but more accurate)
python benchmark_comparison.py 

# Run specific configurations
python benchmark_comparison.py --configs stock hybrid

# Generate more tokens for more accurate results
python benchmark_comparison.py --max-tokens 50
```

The benchmark will produce:
1. A table with performance results for each configuration and batch size
2. A comparative speedup report against stock kernels
3. A visualization plot saved as `benchmark_results.png`

## Configuring Kernel Selection

The kernel selection is automatic by default, but you can manually override it:

```python
# Create model with specific kernel configuration
config = xLSTMLargeConfig(
    embedding_dim=4096,
    num_heads=8,
    num_blocks=32,
    vocab_size=50304,
    # Force specific kernel types
    chunkwise_kernel="chunkwise--native_autograd",
    sequence_kernel="native_sequence__triton",
    step_kernel="native"
)

model = xLSTMLarge(config)
```

## AMD Hardware Detection

The library uses multiple methods to detect AMD hardware:

1. Environment variable detection (`ROCM_PATH`, `HIP_PLATFORM`, etc.)
2. GPU device name inspection (checks for "AMD", "Radeon", "MI", etc.)
3. Specific MI300X detection for targeted optimizations

You can check if AMD hardware was detected in your code:

```python
from mlstm_kernels.triton.amd_detection import is_amd_gpu, is_mi300x

if is_amd_gpu():
    print("AMD GPU detected!")
    
if is_mi300x():
    print("AMD MI300X detected!")
```

## Troubleshooting

If you encounter issues:

1. **Memory errors**: Reduce batch size or sequence length
2. **Weight loading issues**: Check that the model version matches the weights
3. **Kernel selection errors**: Try fallback to native kernels with:
   ```python
   config = xLSTMLargeConfig(
       # ... other parameters ...
       chunkwise_kernel="chunkwise--native_autograd",
       sequence_kernel="native_sequence__native",
       step_kernel="native"
   )
   ```

## Performance Tips

For best performance on AMD hardware:

1. For batch size 1-2: Use hybrid kernels (`chunkwise--native_autograd` + `native_sequence__triton`)
2. For batch size 3-8: Use AMD optimized kernels (`chunkwise--triton_xl_chunk` + `native_sequence__native`)
3. For batch size >8: Use fully native kernels

The library will automatically apply these optimizations based on your input dimensions.

## Additional Resources

- [AMD Optimization Guide](amd_optimization_guide.md): Detailed explanation of the optimizations
- [Benchmark Results](benchmark_results.md): Comparison of different kernel configurations
- [Library Reference](reference.md): Complete API reference 