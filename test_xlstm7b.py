import torch
import time
import os
import argparse
import sys
import importlib
from transformers import AutoModelForCausalLM, AutoTokenizer
from xlstm.xlstm_large.model import xLSTMLargeConfig
from mlstm_kernels.triton.kernel_param_heuristics import is_amd_override, is_amd_hardware
from mlstm_kernels.triton.amd_optimizations import enable_amd_optimizations

def verify_environment():
    """Verify and print environment information"""
    print("\n===== Environment Information =====")
    
    # Check transformers version
    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
        print(f"Transformers path: {transformers.__file__}")
        
        # Check if it's our fork
        is_nx_fork = False
        
        # Method 1: Check for NX-AI in path or version
        if "NX-AI" in transformers.__file__ or "xlstm" in transformers.__version__:
            is_nx_fork = True
        
        # Method 2: Check if the fork has xLSTM-specific classes
        if hasattr(transformers, 'models') and 'xlstm' in dir(transformers.models):
            is_nx_fork = True
            
        # Method 3: Check if the models module has xlstm
        models_dir = dir(transformers.models) if hasattr(transformers, 'models') else []
        if 'xlstm' in models_dir or 'xlstmlarge' in models_dir or 'xLSTM' in models_dir:
            is_nx_fork = True
            
        # Method 4: Check if specific integrations exist
        if hasattr(transformers, 'integrations'):
            integrations_dir = dir(transformers.integrations)
            if 'xlstm' in integrations_dir or 'xLSTM' in integrations_dir:
                is_nx_fork = True
                
        # Method 5: Try importing specific xlstm components
        try:
            from transformers import XLSTMConfig
            is_nx_fork = True
        except ImportError:
            pass
            
        if is_nx_fork:
            print("✅ Using NX-AI transformers fork with xLSTM integration")
        else:
            print("❌ Not using the NX-AI transformers fork")
            print("   Checking available model architectures:")
            # List available model architectures to debug
            if hasattr(transformers, 'models'):
                print(f"   Models: {sorted(dir(transformers.models))[:10]}...")
    except ImportError:
        print("❌ Transformers not installed")
    
    # Check PyTorch version and device
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    
    # Check if HIP is being used
    if 'HIP_VISIBLE_DEVICES' in os.environ:
        print(f"HIP_VISIBLE_DEVICES: {os.environ['HIP_VISIBLE_DEVICES']}")
    
    # Check AMD detection
    is_amd = is_amd_hardware()
    print(f"AMD GPU detected: {is_amd}")
    
    # Print environment variables related to AMD optimizations
    print("AMD optimization environment variables:")
    for var in ["XLSTM_FORCE_STOCK_KERNELS", "DISABLE_AMD_OPTIMIZATIONS", 
                "FORCE_AMD_DETECTION", "AMD_CDNA3_OPTIMIZATIONS", 
                "AMD_PREFER_HYBRID_KERNELS", "XLSTM_OPTIMIZE_BATCH"]:
        if var in os.environ:
            print(f"  {var}={os.environ[var]}")
    
    print("="*40)

def manually_set_kernels(model, kernel_mode):
    """Attempt to manually set kernel configuration in the model"""
    try:
        # First try to locate the model backbone
        if hasattr(model, 'model') and hasattr(model.model, 'model'):
            backbone = model.model.model.backbone
        elif hasattr(model, 'backbone'):
            backbone = model.backbone
        else:
            print("Could not locate model backbone structure")
            return False
        
        # Try to find blocks and set kernel config
        if hasattr(backbone, 'blocks') and len(backbone.blocks) > 0:
            blocks = backbone.blocks
            success = False
            
            for i, block in enumerate(blocks):
                if hasattr(block, 'mlstm_layer') and hasattr(block.mlstm_layer, 'mlstm_backend'):
                    backend = block.mlstm_layer.mlstm_backend
                    
                    if kernel_mode == "stock":
                        backend.config.chunkwise_kernel = "chunkwise--native_autograd"
                        backend.config.sequence_kernel = "native_sequence__native"
                        backend.config.step_kernel = "native"
                    elif kernel_mode == "hybrid":
                        # Use AMD hybrid kernels
                        backend.config.chunkwise_kernel = "chunkwise--triton_xl_chunk"
                        backend.config.sequence_kernel = "native_sequence__triton"
                        backend.config.step_kernel = "triton"
                    
                    print(f"Set block {i} kernels to: {backend.config.chunkwise_kernel}")
                    success = True
            
            return success
    except Exception as e:
        print(f"Error setting kernels: {e}")
    
    return False

def get_optimized_config():
    """Create an optimized xLSTM config for AMD GPUs"""
    config = xLSTMLargeConfig(
        embedding_dim=512,  # Small config for testing
        num_heads=4,
        num_blocks=6,
        vocab_size=2048,
        return_last_states=True,
        mode="inference",
        # AMD optimized kernels
        chunkwise_kernel="chunkwise--triton_xl_chunk",
        sequence_kernel="native_sequence__triton",
        step_kernel="triton"
    )
    return config

def run_benchmark(kernel_mode="auto", num_tokens=100, temperature=0.7, prompt=None, num_runs=3, warmup_tokens=30):
    """
    Run a benchmark with the specified kernel mode.
    
    Args:
        kernel_mode: 'stock', 'hybrid', or 'auto' (uses hardware detection)
        num_tokens: Number of tokens to generate
        temperature: Temperature for generation
        prompt: Text prompt to use
        num_runs: Number of benchmark runs to average
        warmup_tokens: Number of tokens to generate in warmup
    
    Returns:
        Dictionary with generation stats
    """
    # First verify the environment
    verify_environment()
    
    # Use proper HuggingFace model ID
    model_name = "NX-AI/xLSTM-7b"
    
    # Set default prompt if none provided
    if prompt is None:
        prompt = "In a world where technology and nature coexist,"
    
    # Clean up any previous environment variable settings
    for var in ["XLSTM_FORCE_STOCK_KERNELS", "DISABLE_AMD_OPTIMIZATIONS", 
               "FORCE_AMD_DETECTION", "AMD_CDNA3_OPTIMIZATIONS", 
               "AMD_PREFER_HYBRID_KERNELS", "XLSTM_OPTIMIZE_BATCH"]:
        if var in os.environ:
            del os.environ[var]
    
    # Set kernel mode environment variables if needed
    if kernel_mode == "stock":
        print("\n===== Testing with stock (original) kernels =====")
        # Force native implementation regardless of hardware
        os.environ["FORCE_AMD_DETECTION"] = "0"
        os.environ["XLSTM_FORCE_STOCK_KERNELS"] = "1"
        os.environ["DISABLE_AMD_OPTIMIZATIONS"] = "1"
        # Override AMD detection for this run
        is_amd_override(False)
    elif kernel_mode == "hybrid":
        print("\n===== Testing with AMD hybrid-optimized kernels =====")
        # Force AMD optimizations to be used
        os.environ["FORCE_AMD_DETECTION"] = "1"
        os.environ["XLSTM_FORCE_STOCK_KERNELS"] = "0"
        os.environ["DISABLE_AMD_OPTIMIZATIONS"] = "0"
        # Additional optimizations
        os.environ["AMD_CDNA3_OPTIMIZATIONS"] = "1"
        os.environ["AMD_PREFER_HYBRID_KERNELS"] = "1"
        os.environ["XLSTM_OPTIMIZE_BATCH"] = "1"
        # Override AMD detection for this run
        is_amd_override(True)
        # Explicitly enable AMD optimizations
        enable_amd_optimizations()
    else:  # auto
        print("\n===== Testing with automatic kernel detection =====")
        # Use hardware detection (default)
        is_amd_override(None)
    
    # Verify environment after setting variables
    print("Environment variables set for this run:")
    for var in ["XLSTM_FORCE_STOCK_KERNELS", "DISABLE_AMD_OPTIMIZATIONS", 
               "FORCE_AMD_DETECTION", "AMD_CDNA3_OPTIMIZATIONS", 
               "AMD_PREFER_HYBRID_KERNELS", "XLSTM_OPTIMIZE_BATCH"]:
        if var in os.environ:
            print(f"  {var}={os.environ[var]}")
    
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
    
    # Attempt to manually set kernels based on mode
    kernel_set = manually_set_kernels(model, kernel_mode)
    if kernel_set:
        print(f"Successfully set {kernel_mode} kernels manually")
    
    # Get kernel types being used
    try:
        # Try direct access first
        if hasattr(model, 'model') and hasattr(model.model, 'model') and hasattr(model.model.model, 'backbone'):
            backbone = model.model.model.backbone
            if hasattr(backbone, 'blocks') and len(backbone.blocks) > 0:
                block = backbone.blocks[0]
                if hasattr(block, 'mlstm_layer') and hasattr(block.mlstm_layer, 'mlstm_backend'):
                    backend = block.mlstm_layer.mlstm_backend
                    if hasattr(backend, 'config'):
                        config = backend.config
                        kernel_info = f"Using kernel: {config.chunkwise_kernel}"
                    else:
                        kernel_info = "Backend exists but no config found"
                else:
                    kernel_info = "Block exists but no mlstm_layer or backend found"
            else:
                kernel_info = "Backbone exists but no blocks found"
        else:
            # Try to explore the model structure
            print("Exploring model structure for kernel info:")
            found = False
            
            if hasattr(model, 'model'):
                print(" - model.model exists")
                if hasattr(model.model, 'model'):
                    print(" - model.model.model exists")
                    if hasattr(model.model.model, 'backbone'):
                        print(" - model.model.model.backbone exists")
                        if hasattr(model.model.model.backbone, 'blocks'):
                            print(f" - backbone has {len(model.model.model.backbone.blocks)} blocks")
                            
                            # Check the first block
                            if len(model.model.model.backbone.blocks) > 0:
                                block = model.model.model.backbone.blocks[0]
                                print(f" - Block attributes: {dir(block)}")
                                
                                if hasattr(block, 'mlstm_layer'):
                                    print(" - mlstm_layer exists")
                                    print(f" - mlstm_layer attributes: {dir(block.mlstm_layer)}")
                                    
                                    if hasattr(block.mlstm_layer, 'mlstm_backend'):
                                        print(" - mlstm_backend exists")
                                        print(f" - backend attributes: {dir(block.mlstm_layer.mlstm_backend)}")
                                        
                                        if hasattr(block.mlstm_layer.mlstm_backend, 'config'):
                                            config = block.mlstm_layer.mlstm_backend.config
                                            kernel_info = f"Using kernel: {config.chunkwise_kernel}"
                                            found = True
                                        else:
                                            kernel_info = "Backend exists but no config found"
                                    else:
                                        kernel_info = "mlstm_layer exists but no backend found"
                                else:
                                    kernel_info = "Block exists but no mlstm_layer found"
                            else:
                                kernel_info = "Backbone has no blocks"
                        else:
                            kernel_info = "Backbone exists but no blocks attribute"
                    else:
                        kernel_info = "model.model.model exists but no backbone"
                else:
                    kernel_info = "model.model exists but no model.model.model"
            else:
                kernel_info = "No model.model attribute found"
            
            if not found:
                kernel_info = f"Could not determine kernel configuration: {kernel_info}"
    except (AttributeError, IndexError) as e:
        kernel_info = f"Could not determine kernel configuration: {str(e)}"
    
    print(f"Kernel info: {kernel_info}")
    print(f"Prompt: {prompt}")
    
    # Tokenize
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    prompt_length = input_ids.shape[1]
    
    # Extensive warmup phase
    print(f"Performing extensive warmup with {warmup_tokens} tokens...")
    with torch.no_grad():
        # Multiple warmup rounds
        for i in range(3):
            print(f"Warmup round {i+1}/3...")
            _ = model.generate(
                input_ids, 
                max_new_tokens=warmup_tokens, 
                do_sample=True, 
                temperature=temperature
            )
            torch.cuda.synchronize()
            # Brief pause to allow GPU to stabilize
            time.sleep(0.5)
    
    # Multi-run performance measurement
    print(f"Running {num_runs} benchmark iterations...")
    all_times = []
    all_tokens_per_second = []
    
    for run in range(num_runs):
        print(f"Benchmark run {run+1}/{num_runs}...")
        torch.cuda.synchronize()  # Ensure previous operations are complete
        
        # Actual generation with timing
        start_time = time.time()
        output = model.generate(
            input_ids, 
            max_new_tokens=num_tokens,
            do_sample=True,
            temperature=temperature
        )
        torch.cuda.synchronize()  # Ensure generation is complete
        
        end_time = time.time()
        time_taken = end_time - start_time
        tokens_generated = output.shape[1] - prompt_length
        tokens_per_second = tokens_generated / time_taken
        
        all_times.append(time_taken)
        all_tokens_per_second.append(tokens_per_second)
        
        print(f"  Run {run+1}: {tokens_per_second:.2f} tokens/sec ({time_taken:.2f}s)")
        
        # Small pause between runs to reduce thermal effects
        if run < num_runs - 1:
            time.sleep(1.0)
    
    # Calculate statistics
    avg_time = sum(all_times) / len(all_times)
    avg_tokens_per_second = sum(all_tokens_per_second) / len(all_tokens_per_second)
    
    # Calculate variance/std dev if we have multiple runs
    if len(all_tokens_per_second) > 1:
        variance = sum((x - avg_tokens_per_second) ** 2 for x in all_tokens_per_second) / len(all_tokens_per_second)
        std_dev = variance ** 0.5
        std_dev_percent = (std_dev / avg_tokens_per_second) * 100
    else:
        std_dev = 0
        std_dev_percent = 0
    
    # Get the generated text from the last run
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    print(f"\nGenerated text (from final run):")
    print(generated_text)
    
    print(f"\nBenchmark statistics ({num_runs} runs):")
    print(f"Tokens generated per run: {num_tokens}")
    print(f"Average time: {avg_time:.2f} seconds")
    print(f"Average tokens per second: {avg_tokens_per_second:.2f}")
    print(f"Standard deviation: {std_dev:.2f} tokens/sec ({std_dev_percent:.2f}%)")
    print(f"Min: {min(all_tokens_per_second):.2f} tokens/sec")
    print(f"Max: {max(all_tokens_per_second):.2f} tokens/sec")
    
    # Return stats
    return {
        "kernel_mode": kernel_mode,
        "kernel_info": kernel_info,
        "prompt_length": prompt_length,
        "tokens_generated": num_tokens,
        "avg_time_taken": avg_time,
        "avg_tokens_per_second": avg_tokens_per_second,
        "std_dev": std_dev,
        "std_dev_percent": std_dev_percent,
        "min_tokens_per_second": min(all_tokens_per_second),
        "max_tokens_per_second": max(all_tokens_per_second),
        "all_times": all_times,
        "all_tokens_per_second": all_tokens_per_second,
        "load_time": load_time,
        "device": str(device),
        "generated_text": generated_text,
        "num_runs": num_runs
    }

def compare_kernels(num_tokens=100, temperature=0.7, prompt=None, num_runs=3, warmup_tokens=30):
    """Compare stock vs hybrid kernels and print a summary."""
    results = []
    
    # Test stock kernels
    stock_results = run_benchmark("stock", num_tokens, temperature, prompt, num_runs, warmup_tokens)
    results.append(stock_results)
    
    # Clear CUDA cache between runs and cool down period
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("\nCooling down for 3 seconds before next test...")
    time.sleep(3.0)
    
    # Test hybrid kernels
    hybrid_results = run_benchmark("hybrid", num_tokens, temperature, prompt, num_runs, warmup_tokens)
    results.append(hybrid_results)
    
    # Print comparison
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON: STOCK vs HYBRID KERNELS")
    print("="*70)
    
    stock_tps = stock_results["avg_tokens_per_second"]
    hybrid_tps = hybrid_results["avg_tokens_per_second"]
    speedup = (hybrid_tps / stock_tps - 1) * 100
    
    print(f"Stock kernels:  {stock_tps:.2f} tokens/sec (±{stock_results['std_dev_percent']:.2f}%)")
    print(f"Hybrid kernels: {hybrid_tps:.2f} tokens/sec (±{hybrid_results['std_dev_percent']:.2f}%)")
    print(f"Speedup: {speedup:.2f}%")
    
    # Statistical significance check
    if (stock_results['std_dev'] > 0 and hybrid_results['std_dev'] > 0 and 
        abs(stock_tps - hybrid_tps) < (stock_results['std_dev'] + hybrid_results['std_dev'])):
        print("\n⚠️ NOTE: The performance difference may not be statistically significant")
        print(f"The difference ({abs(stock_tps - hybrid_tps):.2f}) is less than the combined standard deviation ({stock_results['std_dev'] + hybrid_results['std_dev']:.2f})")
    
    if speedup < 0:
        print(f"\n⚠️ WARNING: The hybrid kernels are slower than stock kernels!")
        print("This might be due to:")
        print("1. Optimizations not properly applied")
        print("2. Hardware not supported by optimizations")
        print("3. Model configuration issues")
        print("\nCheck the 'Exploring model structure' output above for clues.")
    else:
        print(f"\n✅ Hybrid kernels provide a {speedup:.2f}% speedup!")
        
    # Print run-by-run comparison
    print("\nRun-by-run comparison:")
    print("Run  | Stock (tokens/s) | Hybrid (tokens/s) | Diff (%) ")
    print("-"*55)
    
    for i in range(num_runs):
        stock_run = stock_results["all_tokens_per_second"][i]
        hybrid_run = hybrid_results["all_tokens_per_second"][i]
        run_speedup = (hybrid_run / stock_run - 1) * 100
        print(f"{i+1:4} | {stock_run:16.2f} | {hybrid_run:16.2f} | {run_speedup:+8.2f}%")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test xLSTM model with different kernel configurations')
    parser.add_argument('--mode', type=str, default='compare', choices=['stock', 'hybrid', 'auto', 'compare'],
                        help='Kernel mode to test (stock, hybrid, auto, or compare)')
    parser.add_argument('--tokens', type=int, default=100, help='Number of tokens to generate')
    parser.add_argument('--temp', type=float, default=0.7, help='Temperature for generation')
    parser.add_argument('--prompt', type=str, default=None, help='Text prompt')
    parser.add_argument('--force-amd', action='store_true', help='Force AMD detection for all runs')
    parser.add_argument('--runs', type=int, default=3, help='Number of benchmark runs to average')
    parser.add_argument('--warmup-tokens', type=int, default=30, help='Number of tokens to generate in warmup')
    
    args = parser.parse_args()
    
    # Set global environment variable for AMD detection if requested
    if args.force_amd:
        os.environ["FORCE_AMD_DETECTION"] = "1"
        print("Forcing AMD detection for all runs")
    
    if args.mode == 'compare':
        compare_kernels(args.tokens, args.temp, args.prompt, args.runs, args.warmup_tokens)
    else:
        run_benchmark(args.mode, args.tokens, args.temp, args.prompt, args.runs, args.warmup_tokens) 