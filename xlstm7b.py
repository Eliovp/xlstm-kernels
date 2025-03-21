#!/usr/bin/env python3
# Copyright (c) NXAI GmbH and contributors, AMD.
# Licensed under the NXAI Community License Agreement.

import os
import sys
import time
import torch
import warnings
import argparse
import textwrap
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaAttention

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

# Basic globals for kernel management
_triton_override = None

def is_triton_override(value=None):
    """Set or get the manual triton detection override."""
    global _triton_override
    if value is not None:
        _triton_override = value
    return _triton_override

def is_triton_available():
    """Check if triton kernels are available."""
    global _triton_override
    
    # If there's a manual override, use that
    if _triton_override is not None:
        return _triton_override
        
    # Otherwise actually check for triton kernels
    try:
        # Try to import triton kernels module
        from mlstm_kernels.torch import get_mlstm_kernel
        return True
    except ImportError:
        print("Triton kernels not available")
        return False

def enable_optimizations(kernel_mode):
    """
    Enable optimizations based on kernel mode.
    
    Args:
        kernel_mode: 'stock' or 'triton'
    
    Returns:
        bool: True if optimizations were enabled, False otherwise
    """
    # Set kernel mode environment variables
    if kernel_mode == "stock":
        print("\n===== Using stock (original) kernels =====")
        # Force native implementation
        os.environ["XLSTM_FORCE_STOCK_KERNELS"] = "1"
        return False
        
    elif kernel_mode == "triton":
        print("\n===== Using Triton optimized kernels =====")
        # Try to use triton kernels
        os.environ["XLSTM_FORCE_STOCK_KERNELS"] = "0"
        
        # Check if we have triton kernels
        if is_triton_available():
            print("Using Triton optimized kernels...")
            
            # Load the triton kernels explicitly
            try:
                # Try to import and initialize triton kernels
                import importlib
                try:
                    # First try to ensure the modules are loaded
                    importlib.reload(importlib.import_module('mlstm_kernels.triton.chunkwise'))
                    print("Successfully reloaded Triton kernels")
                except Exception as e:
                    print(f"Couldn't reload triton kernels: {str(e)}")
                    # Try to import anyway
                    import mlstm_kernels.triton.chunkwise
                    
                try:
                    from mlstm_kernels.torch import get_mlstm_kernel
                    print("Successfully loaded mlstm_kernels.torch")
                except Exception as e:
                    print(f"Error loading mlstm_kernels.torch: {str(e)}")
                
                print("Successfully loaded Triton-optimized kernels")
                return True
            except ImportError as e:
                print(f"Warning: Could not load triton kernels: {e}")
                print("Falling back to standard implementation")
                return False
        else:
            print("Triton kernels not available")
            return False
    else:  # auto
        print("\n===== Using automatic kernel detection =====")
        # Use kernel detection (default)
        if is_triton_available():
            print("Triton kernels available, enabling optimizations...")
            os.environ["XLSTM_FORCE_STOCK_KERNELS"] = "0"
            return True
        else:
            print("Using stock kernels based on availability")
            return False

def print_env_info():
    """Print information about the environment."""
    print("\n===== Environment Information =====")
    
    # Print Transformers version
    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
        print(f"Transformers path: {transformers.__file__}")
        
        # Check if Transformers has xLSTM support
        has_xlstm = hasattr(transformers.models, "xlstm")
        if has_xlstm:
            print("✅ Using transformers with xLSTM support")
        else:
            print("⚠️ Warning: Using transformers without xLSTM support")
    except ImportError:
        print("Transformers not installed")
    
    # Print PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    
    # Print CUDA information
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        
    # Print relevant environment variables
    if "HIP_VISIBLE_DEVICES" in os.environ:
        print(f"HIP_VISIBLE_DEVICES: {os.environ['HIP_VISIBLE_DEVICES']}")
    
    # Print current optimization environment variables
    print("Optimization environment variables:")
    optimization_vars = [
        "XLSTM_FORCE_STOCK_KERNELS"
    ]
    
    vars_found = False
    for var in optimization_vars:
        if var in os.environ:
            if not vars_found:
                print("  " + "=" * 40)
                vars_found = True
            print(f"  {var}={os.environ[var]}")
    
    if vars_found:
        print("  " + "=" * 40)
    print()

def load_model_and_tokenizer(model_name="NX-AI/xLSTM-7b", device="cuda"):
    """
    Load a model and tokenizer.
    
    Args:
        model_name: Name or path of the model to load
        device: Device to load the model on
    
    Returns:
        tuple: (model, tokenizer)
    """
    # Load the tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Make sure the tokenizer has a pad token for batch processing
    if tokenizer.pad_token is None:
        print("Setting pad_token to eos_token for batch processing")
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load the model
    print("Loading model...")
    start_time = time.time()
    
    # Load the model with float16 precision
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device
    )
    
    # Print model loading time
    load_time = time.time() - start_time
    print(f"Model loaded on {device} in {load_time:.2f} seconds")
    
    # Set kernels if needed
    if hasattr(model, "set_default_block_mlstm_kernel"):
        kernel_type = "native_autograd" if os.environ.get("XLSTM_FORCE_STOCK_KERNELS") == "1" else "triton_xl_chunk"
        n_blocks = 32  # Should cover most models
        
        for i in range(n_blocks):
            try:
                model.set_default_block_mlstm_kernel(i, f"chunkwise--{kernel_type}")
                print(f"Set block {i} kernels to: chunkwise--{kernel_type}")
            except IndexError:
                # Reached the end of the blocks
                break
        
        if kernel_type == "native_autograd":
            print("Successfully set stock kernels manually")
        else:
            print("Successfully set triton kernels manually")
    
    # Explore the model structure to find more information about kernels
    print("Exploring model structure for kernel info...")
    
    if hasattr(model, "model") and hasattr(model.model, "layers") and len(model.model.layers) > 0:
        # Check if the model has a gated MLP
        if hasattr(model.model.layers[0], "mlp") and hasattr(model.model.layers[0].mlp, "act_fn"):
            print(f"Model activation function: {model.model.layers[0].mlp.act_fn.__class__.__name__}")
        
        # Check for specific kernels or optimized components
        for name, module in model.model.layers[0].named_modules():
            if "mlstm" in name.lower() or "kernel" in name.lower():
                print(f"Found kernel component: {name}")
    else:
        print("Kernel info: Could not find model.model")
    
    return model, tokenizer

def simple_generation(
    model, 
    tokenizer, 
    prompt,
    num_tokens=20,
    temperature=1.0,
    top_p=0.9,
    top_k=40,
    batch_size=1
):
    """
    Generate text from a model.
    
    Args:
        model: The model to generate from
        tokenizer: The tokenizer to use
        prompt: The prompt to generate from
        num_tokens: The number of tokens to generate
        temperature: The temperature to use for sampling
        top_p: The top-p value to use for sampling
        top_k: The top-k value to use for sampling
        batch_size: The batch size to use
        
    Returns:
        tuple: (output_text, tokens_generated, generation_time)
    """
    # Print the prompt
    print(f"Prompt: {prompt}")
    
    # Tokenize the prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    prompt_length = input_ids.shape[-1]
    
    if input_ids.device != model.device:
        input_ids = input_ids.to(model.device)
    
    # Prepare for batch processing if needed
    if batch_size > 1:
        # Duplicate the input for batch processing
        input_ids = input_ids.repeat(batch_size, 1)
    
    # Generate
    start_time = time.time()
    
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_length=prompt_length + num_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Calculate time taken
    generation_time = time.time() - start_time
    
    # Figure out how many tokens were actually generated
    if batch_size > 1:
        if hasattr(output, 'shape'):
            tokens_generated = output.shape[1] - prompt_length
        else:
            try:
                tokens_generated = output[0].shape[1] - prompt_length
            except (IndexError, AttributeError):
                print("Warning: Could not determine exact tokens generated, using requested tokens")
                tokens_generated = num_tokens
    else:
        tokens_generated = output.shape[1] - prompt_length
    
    # Decode the output
    try:
        if batch_size > 1:
            # Use only the first output for display
            output_text = tokenizer.decode(output[0], skip_special_tokens=True)
        else:
            output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Error decoding output: {e}")
        return prompt, 0, generation_time
    
    return output_text, tokens_generated, generation_time

def benchmark_model(
    model,
    tokenizer,
    prompt="Hello, world!",
    num_tokens=100,
    runs=3,
    temperature=0.8,
    warmup_runs=1,
    warmup_tokens=10,
    batch_size=1,
    show_output=True
):
    """
    Benchmark a model's generation speed.
    
    Args:
        model: The model to benchmark
        tokenizer: The tokenizer to use
        prompt: The prompt to generate from
        num_tokens: The number of tokens to generate
        runs: The number of benchmark runs to perform
        temperature: The temperature to use for generation
        warmup_runs: The number of warmup runs to perform
        warmup_tokens: The number of tokens to generate during warmup
        batch_size: The batch size to use
        show_output: Whether to show the generated text
        
    Returns:
        dict: Benchmark results
    """
    # Warm up the model
    if warmup_runs > 0:
        print(f"Performing extensive warmup with {warmup_tokens} tokens...")
        for i in range(warmup_runs):
            print(f"Warmup round {i+1}/{warmup_runs}...")
            _, _, _ = simple_generation(
                model, 
                tokenizer, 
                prompt, 
                num_tokens=warmup_tokens,
                temperature=temperature,
                batch_size=batch_size
            )
    
    # Run the benchmark
    print(f"Running {runs} benchmark iterations...")
    times = []
    tokens_per_second = []
    final_output = None
    
    for i in range(runs):
        print(f"Benchmark run {i+1}/{runs}...")
        output, tokens_generated, generation_time = simple_generation(
            model, 
            tokenizer, 
            prompt, 
            num_tokens=num_tokens,
            temperature=temperature,
            batch_size=batch_size
        )
        
        # Save the output for display
        final_output = output
        
        # Calculate tokens per second
        if generation_time > 0:
            tps = tokens_generated / generation_time
            tokens_per_second.append(tps)
            times.append(generation_time)
            print(f"  Run {i+1}: {tps:.2f} tokens/sec ({generation_time:.2f}s)")
        else:
            print(f"  Run {i+1}: instantaneous generation (too fast to measure)")
    
    # Display the generated text from the final run
    if show_output:
        print(f"\nGenerated text (from final run):")
        print(final_output)
    
    # Calculate statistics
    if tokens_per_second:
        avg_tps = sum(tokens_per_second) / len(tokens_per_second)
        std_tps = np.std(tokens_per_second) if len(tokens_per_second) > 1 else 0
        min_tps = min(tokens_per_second)
        max_tps = max(tokens_per_second)
        avg_time = sum(times) / len(times)
        
        print(f"\nBenchmark statistics ({runs} runs):")
        print(f"Tokens generated per run: {num_tokens}")
        print(f"Average time: {avg_time:.2f} seconds")
        print(f"Average tokens per second: {avg_tps:.2f}")
        print(f"Standard deviation: {std_tps:.2f} tokens/sec ({std_tps/avg_tps*100:.2f}%)")
        print(f"Min: {min_tps:.2f} tokens/sec")
        print(f"Max: {max_tps:.2f} tokens/sec")
        
        return {
            "avg_tps": avg_tps,
            "std_tps": std_tps,
            "relative_std": std_tps / avg_tps,
            "min_tps": min_tps,
            "max_tps": max_tps,
            "runs": runs,
            "tokens": num_tokens,
            "times": times,
            "tokens_per_second": tokens_per_second
        }
    else:
        print("No valid benchmark data collected")
        return None

def compare_kernels(
    model_name="NX-AI/xLSTM-7b",
    prompt="Hello, world!",
    num_tokens=100,
    runs=3,
    temperature=0.8,
    warmup_runs=1,
    warmup_tokens=10,
    batch_size=1,
    show_output=True,
    cooldown_time=3
):
    """
    Compare the performance of different kernel implementations.
    
    Args:
        model_name: The model to use
        prompt: The prompt to generate from
        num_tokens: The number of tokens to generate
        runs: The number of benchmark runs to perform
        temperature: The temperature to use for generation
        warmup_runs: The number of warmup runs to perform
        warmup_tokens: The number of tokens to generate during warmup
        batch_size: The batch size to use
        show_output: Whether to show the generated text
        cooldown_time: Time to wait between tests to cool down hardware
        
    Returns:
        dict: Comparison results
    """
    # Set up results dictionary
    results = {}
    
    # First, run with stock kernels
    print_env_info()
    enable_optimizations(kernel_mode="stock")
    
    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name)
    
    # Run the benchmark
    stock_results = benchmark_model(
        model, 
        tokenizer, 
        prompt=prompt,
        num_tokens=num_tokens,
        runs=runs,
        temperature=temperature,
        warmup_runs=warmup_runs,
        warmup_tokens=warmup_tokens,
        batch_size=batch_size,
        show_output=show_output
    )
    
    # Save the results
    results["stock"] = stock_results
    
    # Cool down
    if cooldown_time > 0:
        print(f"\nCooling down for {cooldown_time} seconds before next test...")
        time.sleep(cooldown_time)
    
    # Then, run with triton kernels
    print_env_info()
    enable_optimizations(kernel_mode="triton")
    
    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name)
    
    # Run the benchmark
    triton_results = benchmark_model(
        model, 
        tokenizer, 
        prompt=prompt,
        num_tokens=num_tokens,
        runs=runs,
        temperature=temperature,
        warmup_runs=warmup_runs,
        warmup_tokens=warmup_tokens,
        batch_size=batch_size,
        show_output=show_output
    )
    
    # Save the results
    results["triton"] = triton_results
    
    # Calculate the speedup
    if stock_results and triton_results:
        stock_tps = stock_results["avg_tps"]
        triton_tps = triton_results["avg_tps"]
        
        speedup = (triton_tps - stock_tps) / stock_tps * 100
        
        print("\n" + "=" * 70)
        print("PERFORMANCE COMPARISON: STOCK vs TRITON KERNELS")
        print("=" * 70)
        print(f"Stock kernels:  {stock_tps:.2f} tokens/sec (±{stock_results['relative_std']*100:.2f}%)")
        print(f"Triton kernels: {triton_tps:.2f} tokens/sec (±{triton_results['relative_std']*100:.2f}%)")
        print(f"Speedup: {speedup:.2f}%")
        
        # Print statistical significance warning if needed
        combined_std = stock_results["std_tps"] + triton_results["std_tps"]
        difference = abs(triton_tps - stock_tps)
        
        if difference < combined_std:
            print("\n⚠️ NOTE: The performance difference may not be statistically significant")
            print(f"The difference ({difference:.2f}) is less than the combined standard deviation ({combined_std:.2f})")
        
        # Print warning if triton is slower
        if speedup < 0:
            print("\n⚠️ WARNING: The triton kernels are slower than stock kernels!")
            print("This might be due to:")
            print("1. Optimizations not properly applied")
            print("2. Hardware not supported by optimizations")
            print("3. Model configuration issues")
            print("\nCheck the 'Exploring model structure' output above for clues.")
        
        # Print run-by-run comparison
        print("\nRun-by-run comparison:")
        print("Run  | Stock (tokens/s) | Triton (tokens/s) | Diff (%)")
        print("-------------------------------------------------------")
        for i in range(min(len(stock_results["tokens_per_second"]), len(triton_results["tokens_per_second"]))):
            stock_run_tps = stock_results["tokens_per_second"][i]
            triton_run_tps = triton_results["tokens_per_second"][i]
            run_diff = (triton_run_tps - stock_run_tps) / stock_run_tps * 100
            print(f"{i+1:4d} | {stock_run_tps:16.2f} | {triton_run_tps:16.2f} | {run_diff:+8.2f}%")
        
        # Save comparison results
        results["speedup"] = speedup
        results["difference"] = difference
        results["combined_std"] = combined_std
        results["statistically_significant"] = difference >= combined_std
    
    return results

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(
        description='xLSTM benchmarking and generation tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''
            Examples:
              # Generate text
              python xlstm7b.py --mode generate --prompt "Once upon a time" --tokens 200
              
              # Run a benchmark
              python xlstm7b.py --mode benchmark --tokens 100 --runs 5
              
              # Compare kernel implementations
              python xlstm7b.py --mode compare --tokens 100 --runs 3
        ''')
    )
    
    # Add common arguments
    parser.add_argument('--model', type=str, default="NX-AI/xLSTM-7b", help='Model name or path')
    parser.add_argument('--tokens', type=int, default=100, help='Number of tokens to generate')
    parser.add_argument('--prompt', type=str, default="Hello, world!", help='Prompt for text generation')
    parser.add_argument('--temp', type=float, default=0.8, help='Temperature for sampling')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for generation')
    parser.add_argument('--device', type=str, default="cuda", help='Device to use (cuda or cpu)')
    
    # Add mode-specific arguments
    parser.add_argument('--mode', type=str, choices=['benchmark', 'generate', 'compare'], default='benchmark', 
                        help='Mode to run (benchmark, generate, or compare)')
    parser.add_argument('--kernel', type=str, choices=['stock', 'triton', 'auto'], default='auto',
                        help='Kernel implementation to use (stock, triton, or auto)')
    parser.add_argument('--runs', type=int, default=3, help='Number of benchmark runs')
    parser.add_argument('--warmup-runs', type=int, default=1, help='Number of warmup runs')
    parser.add_argument('--warmup-tokens', type=int, default=10, help='Number of tokens for warmup')
    parser.add_argument('--cooldown', type=int, default=3, help='Cooldown time between tests (seconds)')
    parser.add_argument('--force-triton', action='store_true', help='Force Triton detection override')
    parser.add_argument('--show-output', type=bool, default=True, help='Show generated text output')
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Force environment variable overrides if specified
    if args.force_triton:
        is_triton_override(True)
    
    # Handle different modes
    if args.mode == "generate":
        # Print environment info
        print_env_info()
        
        # Enable optimizations
        enable_optimizations(kernel_mode=args.kernel)
        
        # Load the model
        model, tokenizer = load_model_and_tokenizer(args.model, device=args.device)
        
        # Generate text
        output, tokens_generated, generation_time = simple_generation(
            model, 
            tokenizer, 
            args.prompt,
            num_tokens=args.tokens,
            temperature=args.temp,
            batch_size=args.batch_size
        )
        
        # Print results
        print(f"\nGenerated {tokens_generated} tokens in {generation_time:.2f} seconds")
        print(f"Tokens per second: {tokens_generated/generation_time:.2f}")
        print("\nGenerated text:")
        print(output)
        
    elif args.mode == "benchmark":
        # Print environment info
        print_env_info()
        
        # Enable optimizations
        enable_optimizations(kernel_mode=args.kernel)
        
        # Load the model
        model, tokenizer = load_model_and_tokenizer(args.model, device=args.device)
        
        # Run the benchmark
        benchmark_model(
            model, 
            tokenizer, 
            prompt=args.prompt,
            num_tokens=args.tokens,
            runs=args.runs,
            temperature=args.temp,
            warmup_runs=args.warmup_runs,
            warmup_tokens=args.warmup_tokens,
            batch_size=args.batch_size,
            show_output=args.show_output
        )
        
    elif args.mode == "compare":
        # Run the comparison
        compare_kernels(
            model_name=args.model,
            prompt=args.prompt,
            num_tokens=args.tokens,
            runs=args.runs,
            temperature=args.temp,
            warmup_runs=args.warmup_runs,
            warmup_tokens=args.warmup_tokens,
            batch_size=args.batch_size,
            show_output=args.show_output,
            cooldown_time=args.cooldown
        )

if __name__ == "__main__":
    main() 