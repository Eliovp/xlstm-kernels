import torch
import time
import os
import sys
import json
from pathlib import Path

# Add installation dir to path
sys.path.append("/app/xlstm-kernels/xlstm")
sys.path.append("/app/xlstm-kernels")

# Import the xLSTM library directly
from xlstm.xlstm_large.model import xLSTMLargeConfig, xLSTMLarge

# Import our AMD batch-aware configuration module
from mlstm_kernels.triton.amd_batch_aware import get_optimal_kernel_config

# Test with a small model (no loading weights) for quick comparison
embedding_dim = 512
num_heads = 4
num_blocks = 6
vocab_size = 2048
head_dim = embedding_dim // num_heads  # For our AMD-specific optimizations

def test_auto_kernel_selection():
    """Test performance using automatic kernel selection based on batch and sequence dimensions"""
    print("\n===== Testing Automatic Kernel Selection =====")
    
    # Define different batch sizes and sequence lengths to test
    test_configs = [
        (1, 2048),  # Batch 1, Seq 2048 - Should use AMD-optimized kernel
        (1, 4096),  # Batch 1, Seq 4096 - Should use native kernel
        (2, 2048),  # Batch 2, Seq 2048 - Should use native kernel
        (4, 2048),  # Batch 4, Seq 2048 - Should use AMD-optimized kernel
    ]
    
    results = []
    
    for batch_size, seq_len in test_configs:
        print(f"\n===== Testing B={batch_size}, S={seq_len} =====")
        
        # Get the optimal kernel configuration based on batch size and sequence length
        kernel_config = get_optimal_kernel_config(batch_size, seq_len, head_dim=head_dim)
        print(f"Selected kernel configuration: {kernel_config}")
        
        # Set up config
        xlstm_config = xLSTMLargeConfig(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            vocab_size=vocab_size,
            return_last_states=True,
            mode="inference",
            **kernel_config
        )
        
        # Create the model instance
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = xLSTMLarge(xlstm_config)
        model = model.to(device)
        
        # Create random inputs
        random_input = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
        print(f"Input shape: {random_input.shape}")
        
        # Warm-up
        with torch.no_grad():
            _ = model(random_input[:, :min(64, seq_len)])
        
        # Run a forward pass
        forward_times = []
        for i in range(5):
            torch.cuda.synchronize()
            start_time = time.time()
            
            with torch.no_grad():
                output = model(random_input)
            
            torch.cuda.synchronize()
            end_time = time.time()
            forward_times.append(end_time - start_time)
        
        # Performance stats for forward passes
        avg_time = sum(forward_times) / len(forward_times)
        tokens_per_sec = (batch_size * seq_len) / avg_time
        
        print(f"Forward pass avg time: {avg_time:.4f} seconds")
        print(f"Tokens per second: {tokens_per_sec:.2f}")
        
        # Test generation performance
        prompt_tokens = torch.randint(0, vocab_size, (1, 10)).to(device)
        max_new_tokens = 20
        
        # Generate tokens
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            # Simple autoregressive generation
            generated = prompt_tokens
            for i in range(max_new_tokens):
                # Get the last tokens that fit in context
                input_chunk = generated[:, -min(seq_len, generated.size(1)):]
                
                # Forward pass
                outputs = model(input_chunk)
                
                # Get the last token's logits
                if isinstance(outputs, tuple):
                    next_token_logits = outputs[0][:, -1, :]
                else:
                    next_token_logits = outputs[:, -1, :]
                
                # Apply temperature and sample
                next_token_logits = next_token_logits / 0.7
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated
                generated = torch.cat([generated, next_token], dim=-1)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        # Generation stats
        gen_time = end_time - start_time
        tokens_per_second = max_new_tokens / gen_time
        
        print(f"Generation time: {gen_time:.4f} seconds")
        print(f"Generation tokens per second: {tokens_per_second:.2f}")
        
        # Store results
        results.append({
            "batch_size": batch_size,
            "seq_len": seq_len,
            "kernel_config": kernel_config,
            "forward_time": avg_time,
            "forward_tokens_per_sec": tokens_per_sec,
            "gen_time": gen_time,
            "gen_tokens_per_sec": tokens_per_second
        })
    
    # Print summary
    print("\n===== PERFORMANCE SUMMARY =====")
    print(f"{'Config':<15} {'Kernel Type':<40} {'Forward Time':<15} {'Forward Tokens/s':<20} {'Gen Time':<15} {'Gen Tokens/s':<15}")
    print("-" * 120)
    
    for result in results:
        config_name = f"B{result['batch_size']}_S{result['seq_len']}"
        kernel_type = result['kernel_config']['chunkwise_kernel']
        print(f"{config_name:<15} {kernel_type:<40} {result['forward_time']:<15.4f} {result['forward_tokens_per_sec']:<20.2f} {result['gen_time']:<15.4f} {result['gen_tokens_per_sec']:<15.2f}")

if __name__ == "__main__":
    test_auto_kernel_selection() 