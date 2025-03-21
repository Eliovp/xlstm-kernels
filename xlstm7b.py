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

def load_model_and_tokenizer(model_name="NousResearch/Nous-Hermes-2-Mixtral-7B-DPO", use_4bit=True, device="cuda"):
    """
    Load a model and tokenizer.
    
    Args:
        model_name: Name or path of the model to load
        use_4bit: Whether to use 4-bit quantization
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
    
    # Set up quantization config if requested
    if use_4bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
    else:
        quantization_config = None
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        quantization_config=quantization_config
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
    Generate text from a prompt.
    
    Args:
        model: The model to use for generation
        tokenizer: The tokenizer to use for encoding and decoding
        prompt: The prompt to generate from
        num_tokens: Number of tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        batch_size: Batch size for generation
    
    Returns:
        tuple: (output_text, tokens_generated, generation_time)
    """
    start_time = time.time()
    device = next(model.parameters()).device
    
    # Tokenize the prompt
    encoding = tokenizer(
        [prompt] * batch_size,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)
    
    prompt_length = encoding.input_ids.shape[1]
    
    # Set up generation parameters
    gen_config = {
        "max_length": prompt_length + num_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": temperature > 0.0001,
        "num_return_sequences": 1
    }
    
    # Generate
    try:
        with torch.no_grad():
            output = model.generate(**encoding, **gen_config)
            
        # Calculate timing
        generation_time = time.time() - start_time
        
        # Count tokens generated (accounting for batch)
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
        
        # Decode the output (only the first in the batch for display)
        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
        
        return decoded_output, tokens_generated, generation_time
        
    except Exception as e:
        print(f"Generation error: {str(e)}")
        return f"Error: {str(e)}", 0, time.time() - start_time

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
    Benchmark generation speed.
    
    Args:
        model: Model to benchmark
        tokenizer: Tokenizer for the model
        prompt: Text prompt for generation
        num_tokens: Number of tokens to generate
        runs: Number of benchmark runs
        temperature: Sampling temperature
        warmup_runs: Number of warmup runs
        warmup_tokens: Number of tokens for warmup
        batch_size: Batch size for generation
        show_output: Whether to show generated output
    
    Returns:
        dict: Dictionary of benchmark results
    """
    # Run warmup iterations
    if warmup_runs > 0:
        print(f"Performing {'extensive ' if warmup_tokens > 20 else ''}warmup with {warmup_tokens} tokens...")
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
    
    # Print benchmark parameters
    print(f"Running {runs} benchmark iterations...")
    
    # Store results
    timings = []
    token_counts = []
    output_text = ""
    
    # Run timed iterations
    for i in range(runs):
        print(f"Benchmark run {i+1}/{runs}...")
        torch.cuda.synchronize()
        
        # Run generation
        output, tokens_generated, generation_time = simple_generation(
            model, 
            tokenizer, 
            prompt,
            num_tokens=num_tokens,
            temperature=temperature,
            batch_size=batch_size
        )
        
        # Print results for this run
        tokens_per_second = tokens_generated / generation_time
        print(f"  Run {i+1}: {tokens_per_second:.2f} tokens/sec ({generation_time:.2f}s)")
        
        # Store results
        timings.append(generation_time)
        token_counts.append(tokens_generated)
        output_text = output
    
    # Print the last generated text if requested
    if show_output:
        print(f"\nGenerated text (from final run):\n{output_text}\n")
    
    # Calculate statistics
    avg_time = np.mean(timings)
    avg_tokens = np.mean(token_counts)
    tokens_per_second = avg_tokens / avg_time
    std_tokens_per_second = np.std([tc/t for tc, t in zip(token_counts, timings)])
    std_percent = (std_tokens_per_second / tokens_per_second) * 100
    
    # Print summary
    print(f"Benchmark statistics ({runs} runs):")
    print(f"Tokens generated per run: {avg_tokens:.0f}")
    print(f"Average time: {avg_time:.2f} seconds")
    print(f"Average tokens per second: {tokens_per_second:.2f}")
    print(f"Standard deviation: {std_tokens_per_second:.2f} tokens/sec ({std_percent:.2f}%)")
    print(f"Min: {min([tc/t for tc, t in zip(token_counts, timings)]):.2f} tokens/sec")
    print(f"Max: {max([tc/t for tc, t in zip(token_counts, timings)]):.2f} tokens/sec")
    
    # Return results
    return {
        "tokens_per_second": tokens_per_second,
        "std_tokens_per_second": std_tokens_per_second,
        "std_percent": std_percent,
        "timings": timings,
        "token_counts": token_counts,
        "output_text": output_text
    }

def compare_kernels(
    model_name="NousResearch/Nous-Hermes-2-Mixtral-7B-DPO",
    prompt="Hello, world!",
    num_tokens=100,
    runs=3,
    temperature=0.8,
    use_4bit=True,
    warmup_runs=1,
    warmup_tokens=10,
    batch_size=1,
    show_output=True,
    cooldown_time=3
):
    """
    Compare stock and Triton kernels performance.
    
    Args:
        model_name: Name or path of the model to load
        prompt: Text prompt for generation
        num_tokens: Number of tokens to generate
        runs: Number of benchmark runs
        temperature: Sampling temperature
        use_4bit: Whether to use 4-bit quantization
        warmup_runs: Number of warmup runs
        warmup_tokens: Number of tokens for warmup
        batch_size: Batch size for generation
        show_output: Whether to show generated output
        cooldown_time: Time to wait between tests to cool down GPU
    
    Returns:
        tuple: (stock_results, triton_results)
    """
    # Run with stock kernels first
    print_env_info()
    enable_optimizations(kernel_mode="stock")
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name, 
        use_4bit=use_4bit
    )
    
    # Run benchmark
    print(f"Prompt: {prompt}")
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
    
    # Cool down between tests
    if cooldown_time > 0:
        print(f"\nCooling down for {cooldown_time} seconds before next test...")
        time.sleep(cooldown_time)
    
    # Delete model to free GPU memory
    del model
    torch.cuda.empty_cache()
    
    # Now run with triton kernels
    print_env_info()
    enable_optimizations(kernel_mode="triton")
    
    # Load model again
    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        use_4bit=use_4bit
    )
    
    # Run benchmark
    print(f"Prompt: {prompt}")
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
    
    # Print comparison
    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON: STOCK vs TRITON KERNELS")
    print("=" * 70)
    
    stock_tps = stock_results["tokens_per_second"]
    triton_tps = triton_results["tokens_per_second"]
    speedup = ((triton_tps / stock_tps) - 1) * 100
    
    stock_std = stock_results["std_tokens_per_second"]
    triton_std = triton_results["std_tokens_per_second"]
    combined_std = stock_std + triton_std
    
    print(f"Stock kernels:  {stock_tps:.2f} tokens/sec (±{stock_results['std_percent']:.2f}%)")
    print(f"Triton kernels: {triton_tps:.2f} tokens/sec (±{triton_results['std_percent']:.2f}%)")
    print(f"Speedup: {speedup:.2f}%")
    
    # Determine if result is statistically significant
    difference = abs(stock_tps - triton_tps)
    if difference < combined_std:
        print("\n⚠️ NOTE: The performance difference may not be statistically significant")
        print(f"The difference ({difference:.2f}) is less than the combined standard deviation ({combined_std:.2f})")
    
    # Print a warning if triton is slower
    if triton_tps < stock_tps:
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
    for i in range(runs):
        stock_run_tps = stock_results["token_counts"][i] / stock_results["timings"][i]
        triton_run_tps = triton_results["token_counts"][i] / triton_results["timings"][i]
        run_speedup = ((triton_run_tps / stock_run_tps) - 1) * 100
        print(f"{i+1:4d} | {stock_run_tps:14.2f} | {triton_run_tps:16.2f} | {run_speedup:+8.2f}%")
    
    return stock_results, triton_results

def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(description="xLSTM Benchmark Tool")
    
    # Add arguments
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["benchmark", "generate", "compare"], 
        default="benchmark",
        help="Run mode: benchmark, generate, or compare kernels"
    )
    parser.add_argument(
        "--kernel", 
        type=str, 
        choices=["stock", "triton", "auto"], 
        default="auto",
        help="Kernel type to use"
    )
    parser.add_argument(
        "--prompt", 
        type=str, 
        default="Hello, world!",
        help="Prompt for generation"
    )
    parser.add_argument(
        "--tokens", 
        type=int, 
        default=100,
        help="Number of tokens to generate"
    )
    parser.add_argument(
        "--runs", 
        type=int, 
        default=3,
        help="Number of benchmark runs"
    )
    parser.add_argument(
        "--temp", 
        type=float, 
        default=0.8,
        help="Temperature for sampling"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="NousResearch/Nous-Hermes-2-Mixtral-7B-DPO",
        help="Model name or path"
    )
    parser.add_argument(
        "--use-4bit", 
        action="store_true",
        help="Use 4-bit quantization"
    )
    parser.add_argument(
        "--warmup-runs", 
        type=int, 
        default=1,
        help="Number of warmup runs"
    )
    parser.add_argument(
        "--warmup-tokens", 
        type=int, 
        default=10,
        help="Number of tokens for warmup"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=1,
        help="Batch size for generation"
    )
    parser.add_argument(
        "--no-output", 
        action="store_true",
        help="Don't show generated output"
    )
    parser.add_argument(
        "--cooldown", 
        type=int, 
        default=3,
        help="Cooldown time in seconds between tests"
    )
    parser.add_argument(
        "--force-triton", 
        action="store_true",
        help="Force using triton kernels"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Apply manual overrides if specified
    if args.force_triton:
        is_triton_override(True)
    
    # Run in the specified mode
    if args.mode == "benchmark":
        # Print environment information
        print_env_info()
        enable_optimizations(kernel_mode=args.kernel)
        
        # Load model
        model, tokenizer = load_model_and_tokenizer(
            model_name=args.model, 
            use_4bit=args.use_4bit
        )
        
        # Run benchmark
        print(f"Prompt: {args.prompt}")
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
            show_output=not args.no_output
        )
    
    elif args.mode == "generate":
        # Print environment information
        print_env_info()
        enable_optimizations(kernel_mode=args.kernel)
        
        # Load model
        model, tokenizer = load_model_and_tokenizer(
            model_name=args.model, 
            use_4bit=args.use_4bit
        )
        
        # Generate
        print(f"Prompt: {args.prompt}")
        output, tokens_generated, generation_time = simple_generation(
            model, 
            tokenizer, 
            args.prompt,
            num_tokens=args.tokens,
            temperature=args.temp,
            batch_size=args.batch_size
        )
        
        # Print results
        tokens_per_second = tokens_generated / generation_time
        print(f"\nGeneration time: {generation_time:.2f} seconds")
        print(f"Tokens generated: {tokens_generated}")
        print(f"Tokens per second: {tokens_per_second:.2f}")
        print(f"\nGenerated text:\n{output}")
    
    elif args.mode == "compare":
        # Compare kernels
        compare_kernels(
            model_name=args.model,
            prompt=args.prompt,
            num_tokens=args.tokens,
            runs=args.runs,
            temperature=args.temp,
            use_4bit=args.use_4bit,
            warmup_runs=args.warmup_runs,
            warmup_tokens=args.warmup_tokens,
            batch_size=args.batch_size,
            show_output=not args.no_output,
            cooldown_time=args.cooldown
        )

if __name__ == "__main__":
    main() 