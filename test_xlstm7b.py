import torch
import time
import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from xlstm.xlstm_large.model import xLSTMLargeConfig
from mlstm_kernels.triton.kernel_param_heuristics import is_amd_override

def run_benchmark(kernel_mode="auto", num_tokens=100, temperature=0.7, prompt=None):
    """
    Run a benchmark with the specified kernel mode.
    
    Args:
        kernel_mode: 'stock', 'hybrid', or 'auto' (uses hardware detection)
        num_tokens: Number of tokens to generate
        temperature: Temperature for generation
        prompt: Text prompt to use
    
    Returns:
        Dictionary with generation stats
    """
    # Use proper HuggingFace model ID
    model_name = "NX-AI/xLSTM-7b"
    
    # Set default prompt if none provided
    if prompt is None:
        prompt = "In a world where technology and nature coexist,"
    
    # Set kernel mode environment variables if needed
    if kernel_mode == "stock":
        print("\n===== Testing with stock (original) kernels =====")
        # Override AMD detection to force stock kernels
        is_amd_override(False)
        os.environ["XLSTM_FORCE_STOCK_KERNELS"] = "1"
    elif kernel_mode == "hybrid":
        print("\n===== Testing with AMD hybrid-optimized kernels =====")
        # Enable AMD detection
        is_amd_override(True)
        os.environ["XLSTM_FORCE_STOCK_KERNELS"] = "0"
    else:  # auto
        print("\n===== Testing with automatic kernel detection =====")
        # Use hardware detection (default)
        is_amd_override(None)
        if "XLSTM_FORCE_STOCK_KERNELS" in os.environ:
            del os.environ["XLSTM_FORCE_STOCK_KERNELS"]
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model
    print("Loading model...")
    start_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    load_time = time.time() - start_load
    
    # Print model info
    device = next(model.parameters()).device
    print(f"Model loaded on {device} in {load_time:.2f} seconds")
    
    # Get kernel types being used
    try:
        config = model.model.model.backbone.blocks[0].mlstm_layer.mlstm_backend.config
        kernel_info = f"Using kernel: {config.chunkwise_kernel}"
    except (AttributeError, IndexError):
        kernel_info = "Could not determine kernel configuration"
    
    print(kernel_info)
    print(f"Prompt: {prompt}")
    
    # Tokenize
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    prompt_length = input_ids.shape[1]
    
    # Generate
    print("Generating...")
    
    with torch.no_grad():
        # Warmup
        _ = model.generate(input_ids, max_new_tokens=10)
        torch.cuda.synchronize()
        
        # Actual generation with timing
        start_time = time.time()
        output = model.generate(
            input_ids, 
            max_new_tokens=num_tokens,
            temperature=temperature
        )
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    # Get generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Print timing info
    tokens_generated = output.shape[1] - prompt_length
    time_taken = end_time - start_time
    tokens_per_second = tokens_generated / time_taken
    
    print(f"\nGenerated text:")
    print(generated_text)
    
    print(f"\nGeneration stats:")
    print(f"Tokens generated: {tokens_generated}")
    print(f"Time taken: {time_taken:.2f} seconds")
    print(f"Tokens per second: {tokens_per_second:.2f}")
    
    # Return stats
    return {
        "kernel_mode": kernel_mode,
        "kernel_info": kernel_info,
        "prompt_length": prompt_length,
        "tokens_generated": tokens_generated,
        "time_taken": time_taken,
        "tokens_per_second": tokens_per_second,
        "load_time": load_time,
        "device": str(device),
        "generated_text": generated_text
    }

def compare_kernels(num_tokens=100, temperature=0.7, prompt=None):
    """Compare stock vs hybrid kernels and print a summary."""
    results = []
    
    # Test stock kernels
    stock_results = run_benchmark("stock", num_tokens, temperature, prompt)
    results.append(stock_results)
    
    # Clear CUDA cache between runs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Test hybrid kernels
    hybrid_results = run_benchmark("hybrid", num_tokens, temperature, prompt)
    results.append(hybrid_results)
    
    # Print comparison
    print("\n" + "="*50)
    print("PERFORMANCE COMPARISON: STOCK vs HYBRID KERNELS")
    print("="*50)
    
    stock_tps = stock_results["tokens_per_second"]
    hybrid_tps = hybrid_results["tokens_per_second"]
    speedup = (hybrid_tps / stock_tps - 1) * 100
    
    print(f"Stock kernels:  {stock_tps:.2f} tokens/sec")
    print(f"Hybrid kernels: {hybrid_tps:.2f} tokens/sec")
    print(f"Speedup: {speedup:.2f}%")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test xLSTM model with different kernel configurations')
    parser.add_argument('--mode', type=str, default='compare', choices=['stock', 'hybrid', 'auto', 'compare'],
                        help='Kernel mode to test (stock, hybrid, auto, or compare)')
    parser.add_argument('--tokens', type=int, default=100, help='Number of tokens to generate')
    parser.add_argument('--temp', type=float, default=0.7, help='Temperature for generation')
    parser.add_argument('--prompt', type=str, default=None, help='Text prompt')
    
    args = parser.parse_args()
    
    if args.mode == 'compare':
        compare_kernels(args.tokens, args.temp, args.prompt)
    else:
        run_benchmark(args.mode, args.tokens, args.temp, args.prompt) 