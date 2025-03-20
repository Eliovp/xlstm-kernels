import torch
import time
import os
import sys
import json
import logging
from pathlib import Path
from safetensors.torch import load_file

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add installation dir to path
sys.path.append("/app/xlstm-kernels/xlstm")
sys.path.append("/app/xlstm-kernels")

# Import AMD detection
from mlstm_kernels.triton.amd_detection import is_amd_gpu, is_mi300x, enable_amd_optimizations
from mlstm_kernels.triton.amd_batch_aware import get_optimal_kernel_config

# Import the xLSTM library directly
from xlstm.xlstm_large.model import xLSTMLargeConfig, xLSTMLarge

# Enable AMD optimizations
is_amd = is_amd_gpu()
is_amd_mi300x = is_mi300x()

if is_amd:
    logger.info(f"AMD GPU detected: MI300X = {is_amd_mi300x}")
    enable_amd_optimizations()
else:
    logger.info("No AMD GPU detected, will use standard kernels")

# Load config from json file directly
logger.info("Loading model config from config.json...")
hf_model_path = os.path.expanduser("~/.cache/huggingface/hub/models--NX-AI--xLSTM-7b/snapshots/8c5476860be2a57eb65634679fd6739d17edea7c")

with open(os.path.join(hf_model_path, "config.json"), "r") as f:
    hf_config_dict = json.load(f)
    
logger.info(f"Config loaded with {len(hf_config_dict)} parameters")

# Extract relevant parameters
embedding_dim = hf_config_dict.get("hidden_size", 4096)
num_heads = hf_config_dict.get("num_attention_heads", 32)
num_blocks = hf_config_dict.get("num_hidden_layers", 32)
vocab_size = hf_config_dict.get("vocab_size", 32000)
head_dim = embedding_dim // num_heads

logger.info(f"Model parameters: embedding_dim={embedding_dim}, num_heads={num_heads}, head_dim={head_dim}, num_blocks={num_blocks}")

# Default kernel configuration
kernel_config = {
    "chunkwise_kernel": "chunkwise--native_autograd",
    "sequence_kernel": "native_sequence__native",
    "step_kernel": "native"
}

# For AMD hardware, get the optimal kernel configuration
if is_amd and is_amd_mi300x:
    # Default config is for batch_size=1, seq_len=2048 initially
    # The model will dynamically adjust based on actual inputs during runtime
    kernel_config = get_optimal_kernel_config(1, 2048, head_dim)
    logger.info(f"Using AMD-optimized kernel configuration: {kernel_config}")

# Set up xLSTM config based on HF config
logger.info("Setting up xLSTM config...")
xlstm_config = xLSTMLargeConfig(
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    num_blocks=num_blocks,
    vocab_size=vocab_size,
    return_last_states=True,
    mode="inference",
    **kernel_config
)

logger.info(f"Config created with kernels: chunkwise={xlstm_config.chunkwise_kernel}, sequence={xlstm_config.sequence_kernel}, step={xlstm_config.step_kernel}")

# Create the model instance
logger.info("Creating xLSTM model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = xLSTMLarge(xlstm_config)

# Load weights from safetensors files
logger.info("Loading model weights from safetensors...")
model_files = sorted([
    f for f in os.listdir(hf_model_path) 
    if f.startswith("model-") and f.endswith(".safetensors")
])
logger.info(f"Found {len(model_files)} model files")

# Initialize a state dict to store all weights
state_dict = {}
for file in model_files:
    file_path = os.path.join(hf_model_path, file)
    logger.info(f"Loading {file}...")
    file_state_dict = load_file(file_path)
    state_dict.update(file_state_dict)

logger.info(f"Loaded {len(state_dict)} parameters")

# Move model to device first (loading large models is faster on GPU)
model = model.to(device)

# Load the state dict
try:
    model.load_state_dict(state_dict)
    logger.info("Model weights loaded successfully!")
except Exception as e:
    logger.error(f"Error loading state dict: {e}")
    
    # Print model parameter names for debugging
    model_keys = set(model.state_dict().keys())
    state_dict_keys = set(state_dict.keys())
    
    missing_keys = model_keys - state_dict_keys
    unexpected_keys = state_dict_keys - model_keys
    
    if missing_keys:
        logger.warning(f"Missing keys: {len(missing_keys)}")
        logger.warning(f"Examples: {list(missing_keys)[:5]}")
    
    if unexpected_keys:
        logger.warning(f"Unexpected keys: {len(unexpected_keys)}")
        logger.warning(f"Examples: {list(unexpected_keys)[:5]}")
    
    # Try partial loading
    logger.info("Attempting to load compatible parameters...")
    compatible_keys = {k: v for k, v in state_dict.items() if k in model_keys}
    model.load_state_dict(compatible_keys, strict=False)
    logger.info(f"Partially loaded {len(compatible_keys)}/{len(model_keys)} parameters")

# Load tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("NX-AI/xLSTM-7b")

# Test with different batch sizes and sequence lengths
def test_generation(prompt, batch_size=1, seq_len=None, max_new_tokens=50, temperature=0.8):
    """
    Test generation performance with the given prompt and configuration.
    
    Args:
        prompt: Text prompt for generation
        batch_size: Batch size (repeated prompt)
        seq_len: Manual sequence length to use (for testing)
        max_new_tokens: Number of tokens to generate
        temperature: Generation temperature
    """
    logger.info(f"\n===== Testing generation with batch_size={batch_size} =====")
    logger.info(f"Prompt: {prompt}")
    
    # Tokenize
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    # Repeat for batch size if needed
    if batch_size > 1:
        input_ids = input_ids.repeat(batch_size, 1)
    
    # If a specific sequence length was provided for testing, pad input
    if seq_len is not None and seq_len > input_ids.shape[1]:
        padding = torch.zeros(batch_size, seq_len - input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device)
        input_ids = torch.cat([input_ids, padding], dim=1)
        logger.info(f"Padded input to sequence length {seq_len}")
    
    logger.info(f"Input shape: {input_ids.shape}")
    
    # Generate
    logger.info("Generating text...")
    start_time = time.time()
    context_length = getattr(model.config, "chunk_size", 64) * 8
    
    with torch.no_grad():
        # Generate step by step
        generated = input_ids
        
        for i in range(max_new_tokens):
            # Get the last chunk that fits in the context
            input_chunk = generated[:, -context_length:]
            
            # Forward pass
            outputs = model(input_chunk)
            
            # Get the last token's logits
            if isinstance(outputs, tuple):
                next_token_logits = outputs[0][:, -1, :]
            else:
                next_token_logits = outputs[:, -1, :]
                
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Sample
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated
            generated = torch.cat([generated, next_token], dim=-1)
            
            # Print progress
            if (i + 1) % 10 == 0:
                logger.info(f"Generated {i + 1}/{max_new_tokens} tokens")
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    # Get generated text for the first item in the batch
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    logger.info("\nGenerated text (first example):")
    logger.info(generated_text)
    
    # Print timing info
    tokens_generated = max_new_tokens * batch_size
    time_taken = end_time - start_time
    tokens_per_second = tokens_generated / time_taken
    
    logger.info(f"\nGeneration stats:")
    logger.info(f"Tokens generated: {tokens_generated}")
    logger.info(f"Time taken: {time_taken:.2f} seconds")
    logger.info(f"Tokens per second: {tokens_per_second:.2f}")
    
    return {
        "batch_size": batch_size,
        "tokens_generated": tokens_generated,
        "time_taken": time_taken,
        "tokens_per_second": tokens_per_second,
        "text": generated_text
    }

# Run tests with different batch sizes
prompt = "Write a short poem about AMD MI300x hardware acceleration:"
results = []

# Test with different batch sizes
results.append(test_generation(prompt, batch_size=1, max_new_tokens=50))
results.append(test_generation(prompt, batch_size=2, max_new_tokens=50))
results.append(test_generation(prompt, batch_size=4, max_new_tokens=50))

# Test with a longer sequence length
long_prompt = "In the world of high-performance computing, where every millisecond counts and efficiency is key, the AMD MI300X accelerator stands as a testament to technological innovation. Designed to push the boundaries of what's possible in AI and scientific computing, this accelerator combines powerful hardware with optimized software to deliver unparalleled performance. " * 3
results.append(test_generation(long_prompt, batch_size=1, max_new_tokens=50))

# Print summary
logger.info("\n===== PERFORMANCE SUMMARY =====")
logger.info(f"{'Batch Size':<15} {'Tokens Generated':<20} {'Time (s)':<15} {'Tokens/s':<15}")
logger.info("-" * 70)
for result in results:
    logger.info(f"{result['batch_size']:<15} {result['tokens_generated']:<20} {result['time_taken']:<15.2f} {result['tokens_per_second']:<15.2f}")

logger.info("\nTest completed successfully!") 