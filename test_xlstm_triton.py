import torch
import time
import os
import sys

# Add installation dir to path
sys.path.append("/app/xlstm-kernels/xlstm")

# Import the xLSTM library directly
from xlstm.xlstm_large.model import xLSTMLargeConfig, xLSTMLarge

print("Setting up xLSTM config with Triton kernels for AMD hardware...")
# Using the suggested configuration
xlstm_config = xLSTMLargeConfig(
    embedding_dim=512,
    num_heads=4,
    num_blocks=6,
    vocab_size=2048,
    return_last_states=True,
    mode="inference",
    chunkwise_kernel="chunkwise--triton_xl_chunk",  # xl_chunk == TFLA kernels
    sequence_kernel="native_sequence__triton",
    step_kernel="triton",
)

print(f"Config created: {xlstm_config}")

# Create the model instance
print("Creating xLSTM model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = xLSTMLarge(xlstm_config)
model = model.to(device)
print(f"Model created. Device: {device}")

# Create random inputs as in the example
print("Creating random inputs...")
batch_size = 3
seq_len = 256
random_input = torch.randint(0, 2048, (batch_size, seq_len)).to(device)
print(f"Input shape: {random_input.shape}")

# Run a forward pass
print("Running forward pass...")
start_time = time.time()

with torch.no_grad():
    output = model(random_input)

end_time = time.time()

# Check output shape
if isinstance(output, tuple):
    main_output = output[0]
    print(f"Output is a tuple with {len(output)} elements")
    print(f"Main output shape: {main_output.shape}")
    expected_shape = (seq_len, 2048)
    matches = main_output.shape[1:] == expected_shape
else:
    main_output = output
    print(f"Output shape: {main_output.shape}")
    expected_shape = (seq_len, 2048)
    matches = main_output.shape[1:] == expected_shape

print(f"Output shape matches expected {expected_shape}: {matches}")

# Performance stats
time_taken = end_time - start_time
print(f"\nPerformance stats:")
print(f"Forward pass time: {time_taken:.4f} seconds")
print(f"Tokens per second: {(batch_size * seq_len) / time_taken:.2f}")

# Try a simple generation example
print("\nTesting generation with fixed vocabulary...")
# Create a simple vocabulary for the test model
simple_vocab = ["[PAD]", "[UNK]"] + [f"token_{i}" for i in range(2046)]
prompt_tokens = [10, 20, 30, 40, 50]  # Some random token IDs within vocab range
input_ids = torch.tensor([prompt_tokens]).to(device)

print(f"Prompt tokens: {prompt_tokens}")

# Simple generate function
def generate_simple(model, input_ids, max_new_tokens=20):
    context_length = 256  # Match the training context length
    generated = input_ids
    
    for i in range(max_new_tokens):
        # Forward pass with the context
        input_chunk = generated[:, -context_length:]
        with torch.no_grad():
            outputs = model(input_chunk)
        
        # Get logits and sample next token
        if isinstance(outputs, tuple):
            logits = outputs[0][:, -1, :]
        else:
            logits = outputs[:, -1, :]
        
        # Just pick the highest probability token for deterministic output
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        
        # Add to generated sequence
        generated = torch.cat([generated, next_token], dim=-1)
        
        # Show progress
        if (i + 1) % 5 == 0:
            print(f"Generated {i+1}/{max_new_tokens} tokens")
    
    return generated

# Generate some tokens
print("Generating tokens...")
start_time = time.time()
generated = generate_simple(model, input_ids, max_new_tokens=20)
end_time = time.time()

# Show results
print(f"Input: {input_ids[0].tolist()}")
print(f"Generated: {generated[0].tolist()}")
print(f"Generation time: {end_time - start_time:.4f} seconds")
print(f"Tokens per second: {20 / (end_time - start_time):.2f}")

print("\nTesting complete!") 