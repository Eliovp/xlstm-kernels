# xLSTM-7B Quick Start Guide

This guide provides step-by-step instructions for testing the xLSTM-7B model with different kernel configurations (stock and AMD-optimized hybrid).

## Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA or ROCm compatible GPU
- 16+ GB GPU memory (for the full model)

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/xlstm-kernels.git
cd xlstm-kernels
```

2. **Install dependencies:**

```bash
pip install -e .
pip install safetensors transformers matplotlib
```

3. **Download the model weights (if not already cached):**

```bash
python -c "from transformers import AutoModel; AutoModel.from_pretrained('NX-AI/xLSTM-7b')"
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

## Next Steps

- Read the [AMD Optimization Guide](amd_optimization_guide.md) for technical details
- See the [Installation Guide](installation_guide.md) for more advanced setup options
- Check the benchmark results to understand optimal configurations for your workload

Happy testing! 