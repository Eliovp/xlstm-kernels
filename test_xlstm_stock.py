import torch
import time
import os
import sys
import json
from pathlib import Path
from safetensors.torch import load_file

# Add installation dir to path
sys.path.append("/app/xlstm-kernels/xlstm")

# Import the xLSTM library directly
from xlstm.xlstm_large.model import xLSTMLargeConfig, xLSTMLarge

# Load config from json file directly
print("Loading model config from config.json...")
hf_model_path = os.path.expanduser("~/.cache/huggingface/hub/models--NX-AI--xLSTM-7b/snapshots/8c5476860be2a57eb65634679fd6739d17edea7c")

with open(os.path.join(hf_model_path, "config.json"), "r") as f:
    hf_config_dict = json.load(f)
    
print(f"Config loaded: {hf_config_dict}")

# Extract relevant parameters
embedding_dim = hf_config_dict.get("hidden_size", 4096)
num_heads = hf_config_dict.get("num_attention_heads", 32)
num_blocks = hf_config_dict.get("num_hidden_layers", 32)
vocab_size = hf_config_dict.get("vocab_size", 32000)

print(f"Extracted params: embedding_dim={embedding_dim}, num_heads={num_heads}, num_blocks={num_blocks}, vocab_size={vocab_size}")

# Set up stock xLSTM config based on HF config
print("Setting up xLSTM config with stock kernels...")
xlstm_config = xLSTMLargeConfig(
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    num_blocks=num_blocks,
    vocab_size=vocab_size,
    return_last_states=True,
    mode="inference",
    # Using native kernels first (stock implementation)
    chunkwise_kernel="chunkwise--native_autograd",
    sequence_kernel="native_sequence__native",
    step_kernel="native",
)

print(f"Config created: {xlstm_config}")

# Create the model instance
print("Creating xLSTM model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = xLSTMLarge(xlstm_config)

# Load weights from safetensors files
print("Loading model weights from safetensors...")
model_files = sorted([
    f for f in os.listdir(hf_model_path) 
    if f.startswith("model-") and f.endswith(".safetensors")
])
print(f"Model files: {model_files}")

# Initialize a state dict to store all weights
state_dict = {}
for file in model_files:
    file_path = os.path.join(hf_model_path, file)
    print(f"Loading {file}...")
    file_state_dict = load_file(file_path)
    state_dict.update(file_state_dict)

print(f"Loaded {len(state_dict)} parameters")

# Move model to device first (loading large models is faster on GPU)
model = model.to(device)

# Load the state dict
try:
    model.load_state_dict(state_dict)
    print("Model weights loaded successfully!")
except Exception as e:
    print(f"Error loading state dict: {e}")
    
    # Print model parameter names for debugging
    model_keys = set(model.state_dict().keys())
    state_dict_keys = set(state_dict.keys())
    
    missing_keys = model_keys - state_dict_keys
    unexpected_keys = state_dict_keys - model_keys
    
    if missing_keys:
        print(f"Missing keys: {len(missing_keys)}")
        print(f"Examples: {list(missing_keys)[:5]}")
    
    if unexpected_keys:
        print(f"Unexpected keys: {len(unexpected_keys)}")
        print(f"Examples: {list(unexpected_keys)[:5]}")
    
    # Try partial loading
    print("Attempting to load compatible parameters...")
    compatible_keys = {k: v for k, v in state_dict.items() if k in model_keys}
    model.load_state_dict(compatible_keys, strict=False)
    print(f"Partially loaded {len(compatible_keys)}/{len(model_keys)} parameters")

# Load tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("NX-AI/xLSTM-7b")

# Test with a real prompt
prompt = "Write a short poem about AMD MI300x hardware acceleration:"
print(f"Prompt: {prompt}")

# Tokenize and move to the same device as the model
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
print(f"Input token count: {input_ids.shape[1]}")

# Simple generate function
def generate(model, input_ids, max_new_tokens=100, temperature=0.7):
    context_length = getattr(model.config, "chunk_size", 64) * 8
    print(f"Using context length: {context_length}")
    
    # Start with input_ids
    generated = input_ids
    
    # Generate one token at a time
    for i in range(max_new_tokens):
        # Get the last chunk that fits in the context
        input_chunk = generated[:, -context_length:]
        
        # Forward pass
        with torch.no_grad():
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
        if (i + 1) % 20 == 0:
            print(f"Generated {i + 1}/{max_new_tokens} tokens")
    
    return generated

# Generate
print("Generating with stock kernels...")
start_time = time.time()

with torch.no_grad():
    output = generate(model, input_ids, max_new_tokens=100, temperature=0.8)
    torch.cuda.synchronize()

end_time = time.time()

# Get generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("\nGenerated text:")
print(generated_text)

# Print timing info
tokens_generated = output.shape[1] - input_ids.shape[1]
time_taken = end_time - start_time
tokens_per_second = tokens_generated / time_taken

print(f"\nGeneration stats (stock kernels):")
print(f"Tokens generated: {tokens_generated}")
print(f"Time taken: {time_taken:.2f} seconds")
print(f"Tokens per second: {tokens_per_second:.2f}") 