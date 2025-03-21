# xLSTM-7B Quick Start Guide

This guide provides step-by-step instructions for testing the xLSTM-7B model with different kernel configurations (stock and AMD-optimized hybrid).

## Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA or ROCm compatible GPU
- 16+ GB GPU memory (for the full model)

## Installation

This project provides AMD-optimized kernels for the xLSTM library. There are several ways to set up the environment:

### Option 1: Use setup.py directly (Recommended)

Our setup.py file has built-in commands to install both packages:

```bash
# Clone the repository
git clone https://github.com/Eliovp/xlstm-kernels.git
cd xlstm-kernels

# Install both packages (xlstm-kernels and the AMD-optimized xLSTM)
python setup.py install

# For development mode:
python setup.py develop
```

### Option 2: Manual Installation

If you prefer to install the packages manually:

```bash
# Clone the repositories
git clone https://github.com/Eliovp/xlstm-kernels.git
cd xlstm-kernels

# Install the kernels package
pip install -e .

# Clone and install the xLSTM library
git clone https://github.com/Eliovp/xlstm.git xlstm_temp
cd xlstm_temp
pip install -e .
cd ..
```

### Option 3: Apply optimizations to an existing xLSTM installation

If you already have the original xLSTM library installed separately and want to apply our optimizations to it:

```bash
# Clone the kernels repository
git clone https://github.com/Eliovp/xlstm-kernels.git
cd xlstm-kernels

# Install just the kernels package
pip install -e .

# Go to your existing xLSTM installation directory
cd /path/to/your/existing/xLSTM

# Apply the AMD optimization patches
cp -r /path/to/xlstm-kernels/mlstm_kernels ./
patch -p1 < /path/to/xlstm-kernels/patches/amd_optimizations.patch

# Reinstall the modified xLSTM
pip install -e .
```

3. **Download the model weights (if not already cached):**

```bash
python -c "from transformers import AutoModel; AutoModel.from_pretrained('NX-AI/xLSTM-7b')"
```

## Project Structure

Our implementation has the following structure:

```
xlstm-kernels/
├── mlstm_kernels/       # AMD-optimized kernels
│   └── triton/
│       ├── amd_detection.py           # AMD hardware detection
│       ├── amd_batch_aware.py         # Batch-aware kernel selection
│       └── kernel_param_heuristics.py # Kernel parameter optimization
├── xlstm/              # Modified xLSTM library with AMD detection
├── docs/               # Documentation
├── test_xlstm_*.py     # Various test scripts
└── benchmark_comparison.py  # Benchmark tool
```

## Testing with Stock Kernels

To test the model with the original stock kernel implementation:

```bash
# Run test script with stock kernels
python test_xlstm_stock.py
```

Alternatively, you can run the model directly in Python:

```python
import torch
from xlstm.xlstm_large.model import xLSTMLargeConfig, xLSTMLarge
from transformers import AutoTokenizer

# Set up configuration with stock kernels
config = xLSTMLargeConfig(
    embedding_dim=4096,
    num_heads=8,
    num_blocks=32,
    vocab_size=50304,
    chunkwise_kernel="chunkwise--triton_xl_chunk",
    sequence_kernel="native_sequence__triton",
    step_kernel="triton"
)

# Create model
model = xLSTMLarge(config)
model = model.to("cuda")  # Move to GPU

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("NX-AI/xLSTM-7b")

# Prepare input
prompt = "Write a short poem about high-performance computing:"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

# Generate text
with torch.no_grad():
    generated = input_ids
    for _ in range(50):
        outputs = model(generated)
        next_token_logits = outputs[0][:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        generated = torch.cat([generated, next_token], dim=-1)

# Decode text
generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
print(generated_text)
```

## Testing with AMD-Optimized Hybrid Kernels

To test the model with the AMD-optimized hybrid kernel implementation:

```bash
# Run test script with hybrid kernels
python test_xlstm_hybrid_full.py
```

For direct Python usage:

```python
import torch
from xlstm.xlstm_large.model import xLSTMLargeConfig, xLSTMLarge
from transformers import AutoTokenizer
from mlstm_kernels.triton.amd_detection import is_amd_gpu, enable_amd_optimizations

# Enable AMD optimizations if on AMD hardware
if is_amd_gpu():
    enable_amd_optimizations()

# Set up configuration with hybrid kernels
config = xLSTMLargeConfig(
    embedding_dim=4096,
    num_heads=8,
    num_blocks=32,
    vocab_size=50304,
    chunkwise_kernel="chunkwise--native_autograd",
    sequence_kernel="native_sequence__triton",
    step_kernel="native"
)

# Create model and continue as above...
model = xLSTMLarge(config)
model = model.to("cuda")
```

## Using Automatic Hardware Detection (Recommended)

The simplest way to use the library with optimal performance on any hardware:

```python
import torch
from xlstm.xlstm_large.model import xLSTMLargeConfig, xLSTMLarge
from transformers import AutoTokenizer

# Default configuration will auto-detect hardware
config = xLSTMLargeConfig(
    embedding_dim=4096,
    num_heads=8,
    num_blocks=32,
    vocab_size=50304
)

# Continue as above...
model = xLSTMLarge(config)
model = model.to("cuda")
```

## Running Performance Benchmark

To compare the performance of different kernel configurations:

```bash
# Quick test with small model
python benchmark_comparison.py --small-model --configs stock hybrid

# Full test with the 7B model
python benchmark_comparison.py --configs stock hybrid fully_native amd_optimized

# Test specific batch sizes
python benchmark_comparison.py --configs stock hybrid --batch-sizes 1 4 8
```

## Troubleshooting

If you encounter any issues:

1. **Out of memory errors**: Try reducing batch size or using a smaller model
   ```bash
   python benchmark_comparison.py --small-model
   ```

2. **Weight loading errors**: Check Hugging Face cache path
   ```bash
   ls ~/.cache/huggingface/hub/models--NX-AI--xLSTM-7b/snapshots/
   ```

3. **Model mismatch**: Ensure library versions match
   ```bash
   pip install --upgrade transformers safetensors
   ```

4. **xLSTM library not found**: Make sure you've installed both packages correctly
   ```bash
   # First check if the packages are installed
   pip list | grep -E "xLSTM|xlstm"
   
   # If the xLSTM library is not installed, reinstall xlstm-kernels which will also install the modified xLSTM
   pip install git+https://github.com/Eliovp/xlstm-kernels.git
   
   # Or you can install the modified xLSTM directly
   pip install git+https://github.com/Eliovp/xlstm.git
   ```

5. **Import errors**: Make sure both libraries are in your Python path
   ```python
   # Test importing both libraries
   python -c "import xlstm; import mlstm_kernels; print('Both libraries imported successfully')"
   ```

## Next Steps

- Read the [AMD Optimization Guide](amd_optimization_guide.md) for technical details
- See the [Installation Guide](installation_guide.md) for more advanced setup options
- Check the benchmark results to understand optimal configurations for your workload

Happy testing! 