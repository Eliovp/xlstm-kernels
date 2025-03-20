import torch
import time
import os
import sys

# Add installation dir to path
sys.path.append("/app/xlstm-kernels/xlstm")

# Import the xLSTM library directly
from xlstm.xlstm_large.model import xLSTMLargeConfig, xLSTMLarge

# Check what parameters the config class accepts
print("Checking available parameters for xLSTMLargeConfig...")
import inspect
config_params = inspect.signature(xLSTMLargeConfig.__init__).parameters
print(f"Available parameters: {list(config_params.keys())}")

print("Setting up xLSTM config with native kernels for AMD compatibility...")
# Using a smaller model size for testing
xlstm_config = xLSTMLargeConfig(
    embedding_dim=1024,  # Reduced size
    num_heads=16,        # Reduced size
    num_blocks=12,       # Reduced size
    vocab_size=32000,
    return_last_states=True,
    mode="inference",
    chunk_size=128,      # Explicitly set chunk size
    chunkwise_kernel="chunkwise--native_autograd",  # no Triton kernels
    sequence_kernel="native_sequence__native",      # no Triton kernels
    step_kernel="native",                           # no Triton kernels
)

print(f"Config created with attributes: {dir(xlstm_config)}")

# Setting model path - using pre-trained model weights
model_path = os.path.expanduser("~/.cache/huggingface/hub/models--NX-AI--xLSTM-7b/snapshots/8c5476860be2a57eb65634679fd6739d17edea7c")
weight_files = [
    os.path.join(model_path, f) 
    for f in os.listdir(model_path) 
    if f.startswith("model-") and f.endswith(".safetensors")
]

print(f"Found model weights: {weight_files}")

# Create the model instance
print("Creating xLSTM model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = xLSTMLarge(xlstm_config)
model = model.to(device)
print(f"Model created. Device: {device}")
print(f"Model config: {model.config}")

# Load a tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("NX-AI/xLSTM-7b")

# Use a more structured prompt
prompt = "Write a short poem about AMD MI300x hardware."
print(f"Prompt: {prompt}")

# Tokenize and move to the same device as the model
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

# Generate function (since we're not using HF's generate)
def generate(model, input_ids, max_new_tokens=100, temperature=0.7):
    context_length = getattr(model.config, "chunk_size", 128) * 8
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
            
        # Check the output structure and update if needed
        if i == 0:
            print(f"Output type: {type(outputs)}")
            if isinstance(outputs, tuple):
                print(f"Outputs tuple length: {len(outputs)}")
                for idx, item in enumerate(outputs):
                    if isinstance(item, torch.Tensor):
                        print(f"  outputs[{idx}] shape: {item.shape}")
        
        # Get the last token's logits
        if isinstance(outputs, tuple):
            # Assuming the first element is the logits
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
            print(f"Generated {i + 1}/{max_new_tokens} tokens")
    
    return generated

# Generate
print("Generating...")
start_time = time.time()

with torch.no_grad():
    # Actual generation with timing
    output = generate(model, input_ids, max_new_tokens=50, temperature=0.8)
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

print(f"\nGeneration stats:")
print(f"Tokens generated: {tokens_generated}")
print(f"Time taken: {time_taken:.2f} seconds")
print(f"Tokens per second: {tokens_per_second:.2f}") 