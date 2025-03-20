import torch
import time
import os
import sys
import json
from pathlib import Path

# Add installation dir to path
sys.path.append("/app/xlstm-kernels/xlstm")

# Import the xLSTM library directly
from xlstm.xlstm_large.model import xLSTMLargeConfig, xLSTMLarge

# Test with a small model (no loading weights) for quick comparison
embedding_dim = 512
num_heads = 4
num_blocks = 6
vocab_size = 2048

def run_test(kernel_type="stock"):
    """Test performance with different kernel types"""
    if kernel_type == "stock":
        # Stock kernels
        kernel_config = {
            "chunkwise_kernel": "chunkwise--native_autograd",
            "sequence_kernel": "native_sequence__native",
            "step_kernel": "native"
        }
        print(f"\n===== Testing stock kernels =====")
    elif kernel_type == "amd":
        # AMD optimized kernels
        kernel_config = {
            "chunkwise_kernel": "chunkwise--triton_xl_chunk",  # With our AMD optimizations
            "sequence_kernel": "native_sequence__triton",      # With our AMD optimizations
            "step_kernel": "triton"                            # With our AMD optimizations
        }
        print(f"\n===== Testing AMD-optimized kernels =====")
    elif kernel_type == "hybrid":
        # Hybrid approach - native kernels with our AMD optimizations
        kernel_config = {
            "chunkwise_kernel": "chunkwise--native_autograd",  # Native with AMD optimizations
            "sequence_kernel": "native_sequence__native",      # Native with AMD optimizations
            "step_kernel": "native"                            # Native with AMD optimizations
        }
        print(f"\n===== Testing hybrid kernels (native + AMD opts) =====")
    else:
        # Default to stock
        kernel_config = {
            "chunkwise_kernel": "chunkwise--native_autograd",
            "sequence_kernel": "native_sequence__native",
            "step_kernel": "native"
        }
        print(f"\n===== Testing default kernels =====")
    
    # Set up config
    print(f"Setting up xLSTM config with {kernel_type} kernels...")
    xlstm_config = xLSTMLargeConfig(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_blocks=num_blocks,
        vocab_size=vocab_size,
        return_last_states=True,
        mode="inference",
        **kernel_config
    )
    
    print(f"Config created: {xlstm_config}")
    
    # Create the model instance
    print("Creating xLSTM model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = xLSTMLarge(xlstm_config)
    model = model.to(device)
    print(f"Model created. Device: {device}")
    
    # Create random inputs
    print("Creating random inputs...")
    batch_size = 4
    seq_len = 512
    random_input = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    print(f"Input shape: {random_input.shape}")
    
    # Warm-up
    print("Warm-up pass...")
    with torch.no_grad():
        _ = model(random_input[:, :64])
    
    # Run a forward pass
    print("Running forward pass...")
    forward_times = []
    
    # Run 5 iterations for consistent measurement
    for i in range(5):
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            output = model(random_input)
        
        torch.cuda.synchronize()
        end_time = time.time()
        forward_times.append(end_time - start_time)
    
    # Check output shape
    if isinstance(output, tuple):
        main_output = output[0]
        print(f"Output is a tuple with {len(output)} elements")
        print(f"Main output shape: {main_output.shape}")
    else:
        main_output = output
        print(f"Output shape: {main_output.shape}")
    
    # Performance stats for forward passes
    avg_time = sum(forward_times) / len(forward_times)
    tokens_per_sec = (batch_size * seq_len) / avg_time
    
    print(f"\nForward pass performance stats ({kernel_type} kernels):")
    print(f"Average time over {len(forward_times)} runs: {avg_time:.4f} seconds")
    print(f"Tokens per second: {tokens_per_sec:.2f}")
    
    # Test generation performance
    prompt_tokens = torch.randint(0, vocab_size, (1, 10)).to(device)
    max_new_tokens = 50
    
    # Simple generate function
    def generate(model, input_ids, max_new_tokens=50, temperature=0.7):
        context_length = seq_len  # Use the same as our test sequence length
        generated = input_ids
        
        torch.cuda.synchronize()
        start_time = time.time()
        
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
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        return generated, end_time - start_time
    
    # Run generation
    print("\nTesting generation performance...")
    generated, gen_time = generate(model, prompt_tokens)
    
    # Print generation stats
    tokens_generated = max_new_tokens
    tokens_per_second = tokens_generated / gen_time
    
    print(f"\nGeneration stats ({kernel_type} kernels):")
    print(f"Tokens generated: {tokens_generated}")
    print(f"Time taken: {gen_time:.4f} seconds")
    print(f"Tokens per second: {tokens_per_second:.2f}")
    
    return {
        "kernel_type": kernel_type,
        "forward_time": avg_time,
        "forward_tokens_per_sec": tokens_per_sec,
        "gen_time": gen_time,
        "gen_tokens_per_sec": tokens_per_second
    }

# Run all tests
results = []
results.append(run_test("stock"))      # Stock kernels
results.append(run_test("amd"))        # AMD-optimized Triton kernels
results.append(run_test("hybrid"))     # Native kernels with AMD optimizations

# Print comparison
print("\n===== PERFORMANCE COMPARISON =====")
print(f"{'Kernel Type':<15} {'Forward Time':<15} {'Forward Tokens/s':<20} {'Gen Time':<15} {'Gen Tokens/s':<15}")
print("-" * 80)
for result in results:
    print(f"{result['kernel_type']:<15} {result['forward_time']:<15.4f} {result['forward_tokens_per_sec']:<20.2f} {result['gen_time']:<15.4f} {result['gen_tokens_per_sec']:<15.2f}")

# Calculate and print speedups relative to stock
if len(results) >= 2:
    stock = results[0]
    amd = results[1]
    
    print("\n===== SPEEDUP RELATIVE TO STOCK =====")
    for result in results[1:]:
        kernel_type = result["kernel_type"]
        forward_speedup = result['forward_tokens_per_sec'] / stock['forward_tokens_per_sec']
        gen_speedup = result['gen_tokens_per_sec'] / stock['gen_tokens_per_sec']
        
        print(f"{kernel_type} forward speedup: {forward_speedup:.2f}x")
        print(f"{kernel_type} generation speedup: {gen_speedup:.2f}x") 