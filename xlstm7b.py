#!/usr/bin/env python3
# Copyright (c) NXAI GmbH and contributors, AMD.
# Licensed under the NXAI Community License Agreement.

import os
import sys
import time
import torch
import argparse
import textwrap
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import signal

from transformers import AutoModelForCausalLM, AutoTokenizer

# Check if AMD optimizations are available
try:
    from mlstm_kernels.triton.amd_optimizations import (
        is_amd, is_cdna3, enable_amd_optimizations, get_hip_device_count
    )
    AMD_SUPPORT = True
except ImportError:
    AMD_SUPPORT = False
    print("WARNING: AMD optimizations module could not be imported")
    
    def is_amd():
        """Fallback for AMD detection"""
        # Try alternative methods to detect AMD GPUs
        try:
            import torch
            if hasattr(torch, '_C') and hasattr(torch._C, '_TORCH_CUDA_VERSION'):
                # Check if this is ROCm/HIP build of PyTorch
                if 'rocm' in torch.__version__.lower() or 'hip' in torch.__version__.lower():
                    return True
            
            # Check device name for AMD indicators
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0).lower()
                if any(name in device_name for name in ['amd', 'instinct', 'mi', 'cdna', 'radeon']):
                    return True
        except:
            pass
        
        # Check environment variables
        import os
        if os.environ.get("FORCE_AMD_DETECTION") == "1":
            print("AMD detection forced via environment variable")
            return True
        
        return False
    
    def is_cdna3():
        """Fallback for CDNA3 architecture detection"""
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0).lower()
                if 'mi300' in device_name or 'mi200' in device_name:
                    return True
        except:
            pass
        return False
    
    def enable_amd_optimizations():
        """Fallback for enabling AMD optimizations"""
        print("AMD optimizations not available - using standard implementation")
        
    def get_hip_device_count():
        """Fallback for HIP device count"""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.device_count()
        except:
            pass
        return 0

# Global AMD detection override
_amd_override = None

def is_amd_override(override=None):
    """
    Override AMD detection for testing purposes.
    
    Args:
        override: If None, resets to hardware detection
                 If True, forces detection to return True
                 If False, forces detection to return False
    """
    global _amd_override
    _amd_override = override

def enable_optimizations(kernel_mode):
    """
    Enable optimizations based on kernel mode and hardware detection.
    
    Args:
        kernel_mode: 'stock', 'hybrid', or 'auto'
    
    Returns:
        bool: True if optimizations were enabled, False otherwise
    """
    # Set kernel mode environment variables if needed
    if kernel_mode == "stock":
        print("\n===== Using stock (original) kernels =====")
        # Force native implementation regardless of hardware
        os.environ["FORCE_AMD_DETECTION"] = "0"
        os.environ["XLSTM_FORCE_STOCK_KERNELS"] = "1"
        os.environ["DISABLE_AMD_OPTIMIZATIONS"] = "1"
        # Override AMD detection for this run
        is_amd_override(False)
        return False
        
    elif kernel_mode == "hybrid":
        print("\n===== Using AMD hybrid-optimized kernels =====")
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
        
        # Explicitly check if we have the optimization module
        if AMD_SUPPORT:
            print("Enabling AMD optimizations...")
            enable_amd_optimizations()
            
            # Load the triton kernels explicitly
            try:
                # Try to import and initialize triton kernels
                import mlstm_kernels.triton.chunkwise
                from mlstm_kernels.torch import get_mlstm_kernel
                print("Successfully loaded AMD-optimized kernels")
                return True
            except ImportError as e:
                print(f"Warning: Could not load triton kernels: {e}")
                print("Falling back to standard implementation")
                return False
        else:
            print("AMD optimization module not available")
            return False
            
    else:  # auto
        print("\n===== Using automatic kernel detection =====")
        # Use hardware detection (default)
        is_amd_override(None)
        
        # Detect hardware and enable optimizations if appropriate
        hardware_is_amd = is_amd() if _amd_override is None else _amd_override
        if hardware_is_amd and AMD_SUPPORT:
            print("AMD GPU detected, enabling optimizations...")
            os.environ["XLSTM_FORCE_STOCK_KERNELS"] = "0"
            os.environ["DISABLE_AMD_OPTIMIZATIONS"] = "0"
            os.environ["AMD_CDNA3_OPTIMIZATIONS"] = "1"
            os.environ["AMD_PREFER_HYBRID_KERNELS"] = "1"
            enable_amd_optimizations()
            return True
        else:
            print("Using stock kernels based on hardware detection")
            return False

def manually_set_kernels(model, kernel_mode="auto"):
    """
    Manually set the kernels for the model.
    
    Args:
        model: The loaded model
        kernel_mode: 'stock', 'hybrid', or 'auto'
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Find model backbone (model structure may vary)
        if hasattr(model, 'model') and hasattr(model.model, 'model') and hasattr(model.model.model, 'backbone'):
            backbone = model.model.model.backbone
        else:
            # Try alternative path
            backbone = model
            for attr in ['model', 'backbone']:
                if hasattr(backbone, attr):
                    backbone = getattr(backbone, attr)
        
        if not hasattr(backbone, 'blocks'):
            print("Could not find model blocks")
            return False
        
        # Get the appropriate kernel type
        if kernel_mode == "stock":
            kernel_type = "chunkwise--native_autograd"
        elif kernel_mode == "hybrid":
            kernel_type = "chunkwise--triton_xl_chunk"
        else:
            # Auto - use hardware detection
            hardware_is_amd = is_amd() if _amd_override is None else _amd_override
            if hardware_is_amd:
                kernel_type = "chunkwise--triton_xl_chunk"
            else:
                kernel_type = "chunkwise--native_autograd"
        
        # Apply to all blocks
        for i, block in enumerate(backbone.blocks):
            if not hasattr(block, 'mlstm_layer') or not hasattr(block.mlstm_layer, 'mlstm_backend'):
                print(f"Block {i} does not have mlstm_layer or backend")
                continue
            
            backend = block.mlstm_layer.mlstm_backend
            if not hasattr(backend, 'config'):
                print(f"Block {i} backend does not have config")
                continue
            
            # Set kernel type
            backend.config.chunkwise_kernel = kernel_type
            print(f"Set block {i} kernels to: {kernel_type}")
        
        return True
    except (AttributeError, IndexError) as e:
        print(f"Error setting kernels: {str(e)}")
        return False

def verify_environment():
    """Verify and print environment information."""
    print("\n===== Environment Information =====")
    # Check transformers version
    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
        print(f"Transformers path: {transformers.__file__}")
        
        # Try to check if this is the correct fork by looking for xLSTM functionality
        has_xlstm = False
        try:
            # Check if model type exists
            if hasattr(transformers, 'XLSTMConfig') or 'xlstm' in dir(transformers.models):
                has_xlstm = True
            # Check if we can load the model
            elif "NX-AI/xLSTM-7b" in transformers.models.auto.modeling_auto._MODEL_MAPPING_NAMES.values():
                has_xlstm = True
            # Check if "NX-AI" is in the path
            elif "NX-AI" in transformers.__file__ or "nx-ai" in transformers.__file__:
                has_xlstm = True
            
            if has_xlstm:
                print("✅ Using transformers with xLSTM support")
            else:
                # If we're still here and kernels work, assume it's supported
                print("⚠️ Could not explicitly verify xLSTM support, but will proceed")
        except:
            # If kernels can be set manually, we'll assume this is the right fork
            print("⚠️ Could not verify transformers fork, but will attempt to use xLSTM")
            
    except ImportError:
        print("❌ Transformers not installed")
    
    # Check PyTorch version
    try:
        print(f"PyTorch version: {torch.__version__}")
    except:
        print("❌ PyTorch not installed")
    
    # Check CUDA
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    
    # Check AMD
    hip_visible_devices = os.environ.get("HIP_VISIBLE_DEVICES", "Not set")
    print(f"HIP_VISIBLE_DEVICES: {hip_visible_devices}")
    
    hardware_is_amd = is_amd() if _amd_override is None else _amd_override
    print(f"AMD GPU detected: {hardware_is_amd}")
    
    # Check environment variables
    amd_vars = [
        "XLSTM_FORCE_STOCK_KERNELS", "DISABLE_AMD_OPTIMIZATIONS",
        "FORCE_AMD_DETECTION", "AMD_CDNA3_OPTIMIZATIONS", 
        "AMD_PREFER_HYBRID_KERNELS", "XLSTM_OPTIMIZE_BATCH"
    ]
    
    print("AMD optimization environment variables:")
    for var in amd_vars:
        if var in os.environ:
            print(f"  {var}={os.environ[var]}")
    
    print("=" * 40)

def load_model_and_tokenizer(model_name, device_map=None):
    """
    Load the model and tokenizer with proper error handling.
    
    Args:
        model_name: HuggingFace model name/path
        device_map: Device mapping strategy (auto, balanced, single device, etc.)
        
    Returns:
        Tuple of (model, tokenizer, device)
    """
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Ensure the tokenizer has a padding token for batch processing
    if tokenizer.pad_token is None:
        print("Setting pad_token to eos_token for batch processing")
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading model...")
    start_load = time.time()
    
    # For batch processing, we need to avoid model sharding
    # If batch_size > 1, force the model onto a single device
    if device_map is None:
        # Default for batched operation is to use a single device
        device_map = "auto"
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device_map
        )
        load_time = time.time() - start_load
        
        # Get the device of the first parameter
        device = next(model.parameters()).device
        print(f"Model loaded on {device} in {load_time:.2f} seconds")
        
        # Check if model is split across devices
        devices = {param.device for param in model.parameters()}
        if len(devices) > 1:
            print(f"Note: Model is distributed across {len(devices)} devices: {devices}")
        
        return model, tokenizer, device, load_time
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def simple_generation(kernel_mode="auto", num_tokens=100, temperature=0.7, prompt=None, batch_size=1):
    """
    Simple text generation without benchmarking.
    
    Args:
        kernel_mode: 'stock', 'hybrid', or 'auto' (uses hardware detection)
        num_tokens: Number of tokens to generate
        temperature: Temperature for generation
        prompt: Text prompt to use
        batch_size: Batch size for generation
    
    Returns:
        Generated text
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
    
    # Enable optimizations based on kernel mode
    optimizations_enabled = enable_optimizations(kernel_mode)
    
    # Verify environment after setting variables
    print("Environment variables set for this run:")
    for var in ["XLSTM_FORCE_STOCK_KERNELS", "DISABLE_AMD_OPTIMIZATIONS", 
               "FORCE_AMD_DETECTION", "AMD_CDNA3_OPTIMIZATIONS", 
               "AMD_PREFER_HYBRID_KERNELS", "XLSTM_OPTIMIZE_BATCH"]:
        if var in os.environ:
            print(f"  {var}={os.environ[var]}")
    
    # For batch processing, we need to avoid model sharding
    # If using batches, force the model onto a single device
    device_map = "cuda:0" if batch_size > 1 and torch.cuda.is_available() else "auto"
    if batch_size > 1:
        print(f"Using batch processing with size {batch_size}, forcing model to single device")
    
    # Load model and tokenizer
    try:
        model, tokenizer, device, load_time = load_model_and_tokenizer(model_name, device_map)
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        return None
    
    # Attempt to manually set kernels based on mode
    kernel_set = manually_set_kernels(model, kernel_mode)
    if kernel_set:
        print(f"Successfully set {kernel_mode} kernels manually")
    
    print(f"Prompt: {prompt}")
    
    # Handle batched inputs if needed
    if batch_size > 1:
        print(f"Using batch size: {batch_size}")
        # Repeat the prompt for batched generation
        prompts = [prompt] * batch_size
        # Tokenize with padding
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        
        # Ensure inputs are on the same device as the model
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
    else:
        # Single prompt processing
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        attention_mask = None
    
    # Generate
    print(f"Generating {num_tokens} tokens with temperature {temperature}...")
    start_time = time.time()
    
    try:
        output = model.generate(
            input_ids, 
            attention_mask=attention_mask,
            max_new_tokens=num_tokens,
            do_sample=True,
            temperature=temperature
        )
        
        end_time = time.time()
        time_taken = end_time - start_time
        
        # Get the generated text
        if batch_size > 1:
            try:
                generated_texts = [tokenizer.decode(out, skip_special_tokens=True) for out in output]
            except Exception as e:
                print(f"Error decoding output: {str(e)}")
                # Try alternative decoding if output format is unexpected
                if hasattr(output, "shape") and len(output.shape) == 2:
                    generated_texts = [tokenizer.decode(output[0], skip_special_tokens=True)]
                else:
                    generated_texts = ["Error: Could not decode output"]
        else:
            generated_texts = [tokenizer.decode(output[0], skip_special_tokens=True)]
        
        # Print results
        print(f"\nGeneration completed in {time_taken:.2f} seconds")
        print(f"Average speed: {num_tokens / time_taken:.2f} tokens/sec")
        
        print("\nGenerated text:")
        for i, text in enumerate(generated_texts):
            if batch_size > 1:
                print(f"\n--- Sample {i+1}/{batch_size} ---")
            print(text)
        
        return generated_texts
        
    except Exception as e:
        print(f"An error occurred during generation: {str(e)}")
        print(f"Model device: {device}")
        if batch_size > 1:
            print(f"Input device: {input_ids.device}")
            print(f"Attention mask device: {attention_mask.device}")
        else:
            print(f"Input device: {input_ids.device}")
            
        # Suggest a solution
        print("\nSuggested fix: Try running with batch_size=1 or specify a single GPU with HIP_VISIBLE_DEVICES")
        return None

def run_benchmark(kernel_mode="auto", num_tokens=100, temperature=0.7, prompt=None, num_runs=3, warmup_tokens=30, batch_size=1):
    """
    Run a benchmark with the specified kernel mode.
    
    Args:
        kernel_mode: 'stock', 'hybrid', or 'auto' (uses hardware detection)
        num_tokens: Number of tokens to generate
        temperature: Temperature for generation
        prompt: Text prompt to use
        num_runs: Number of benchmark runs to average
        warmup_tokens: Number of tokens to generate in warmup
        batch_size: Batch size for generation
    
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
    
    # Enable optimizations based on kernel mode
    optimizations_enabled = enable_optimizations(kernel_mode)
    if kernel_mode == "hybrid" and not optimizations_enabled:
        print("WARNING: Hybrid mode selected but optimizations could not be enabled.")
        print("Performance may be suboptimal.")
    
    # Verify environment after setting variables
    print("Environment variables set for this run:")
    for var in ["XLSTM_FORCE_STOCK_KERNELS", "DISABLE_AMD_OPTIMIZATIONS", 
               "FORCE_AMD_DETECTION", "AMD_CDNA3_OPTIMIZATIONS", 
               "AMD_PREFER_HYBRID_KERNELS", "XLSTM_OPTIMIZE_BATCH"]:
        if var in os.environ:
            print(f"  {var}={os.environ[var]}")
    
    # For batch processing, we need to avoid model sharding
    # If using batches, force the model onto a single device
    device_map = "cuda:0" if batch_size > 1 and torch.cuda.is_available() else "auto"
    if batch_size > 1:
        print(f"Using batch processing with size {batch_size}, forcing model to single device")
    
    # Load model and tokenizer
    try:
        model, tokenizer, device, load_time = load_model_and_tokenizer(model_name, device_map)
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        return {
            "kernel_mode": kernel_mode,
            "error": str(e)
        }
    
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
            print("Exploring model structure for kernel info...")
            if hasattr(model, 'model'):
                print(" - Found model.model")
                if hasattr(model.model, 'layers'):
                    print(f" - Found model.model.layers with {len(model.model.layers)} layers")
                    if len(model.model.layers) > 0:
                        layer = model.model.layers[0]
                        print(f" - First layer attributes: {[attr for attr in dir(layer) if not attr.startswith('_')]}")
                        if hasattr(layer, 'mlstm'):
                            print(" - Found layer.mlstm")
                            mlstm = layer.mlstm
                            print(f" - mlstm attributes: {[attr for attr in dir(mlstm) if not attr.startswith('_')]}")
                            if hasattr(mlstm, 'kernel_fn'):
                                kernel_info = f"Using kernel function: {mlstm.kernel_fn}"
                                found = True
                            else:
                                kernel_info = "Could not find kernel_fn in mlstm"
                        else:
                            kernel_info = "Could not find mlstm in layer"
                    else:
                        kernel_info = "model.model.layers is empty"
                else:
                    kernel_info = "Could not find model.model.layers"
            else:
                kernel_info = "Could not find model.model"
    except (AttributeError, IndexError) as e:
        kernel_info = f"Could not determine kernel configuration: {str(e)}"
    
    print(f"Kernel info: {kernel_info}")
    print(f"Prompt: {prompt}")
    
    # Handle batched inputs if needed
    if batch_size > 1:
        print(f"Using batch size: {batch_size}")
        # Repeat the prompt for batched generation
        prompts = [prompt] * batch_size
        # Tokenize with padding
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        prompt_length = input_ids.shape[1]
    else:
        # Single prompt processing
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        attention_mask = None
        prompt_length = input_ids.shape[1]
    
    try:
        # Extensive warmup phase
        print(f"Performing extensive warmup with {warmup_tokens} tokens...")
        with torch.no_grad():
            # Multiple warmup rounds
            for i in range(3):
                print(f"Warmup round {i+1}/3...")
                _ = model.generate(
                    input_ids, 
                    attention_mask=attention_mask,
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
        outputs = []
        
        for run in range(num_runs):
            print(f"Benchmark run {run+1}/{num_runs}...")
            torch.cuda.synchronize()  # Ensure previous operations are complete
            
            # Actual generation with timing
            start_time = time.time()
            output = model.generate(
                input_ids, 
                attention_mask=attention_mask,
                max_new_tokens=num_tokens,
                do_sample=True,
                temperature=temperature
            )
            torch.cuda.synchronize()  # Ensure generation is complete
            outputs.append(output)
            
            end_time = time.time()
            time_taken = end_time - start_time
            
            # Calculate per-batch tokens generated
            if batch_size > 1:
                # For batched output, we need to be careful with the shape handling
                if hasattr(output, 'shape'):
                    # If output is a single tensor with batch dimension
                    tokens_generated = output.shape[1] - prompt_length
                else:
                    # Try to get length of first sequence in batch if output is a list/tuple
                    try:
                        tokens_generated = output[0].shape[1] - prompt_length
                    except (IndexError, AttributeError):
                        # If all else fails, just use the requested tokens
                        print("Warning: Could not determine exact tokens generated, using requested tokens")
                        tokens_generated = num_tokens
            else:
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
        if batch_size > 1:
            try:
                if hasattr(outputs[-1], 'shape'):
                    # Single tensor with batch dimension
                    generated_texts = [tokenizer.decode(outputs[-1][i], skip_special_tokens=True) 
                                      for i in range(min(batch_size, outputs[-1].shape[0]))]
                elif isinstance(outputs[-1], (list, tuple)):
                    # List of tensors
                    generated_texts = [tokenizer.decode(out, skip_special_tokens=True) 
                                      for out in outputs[-1][:batch_size]]
                else:
                    # Fallback
                    generated_texts = ["Could not decode output due to unexpected format"]
            except Exception as e:
                print(f"Warning: Could not decode batched output: {str(e)}")
                generated_texts = [f"Error decoding output: {str(e)}"]
        else:
            try:
                generated_texts = [tokenizer.decode(outputs[-1][0], skip_special_tokens=True)]
            except Exception as e:
                print(f"Warning: Could not decode output: {str(e)}")
                generated_texts = [f"Error decoding output: {str(e)}"]
        
        print(f"\nGenerated text (from final run):")
        for i, text in enumerate(generated_texts):
            if batch_size > 1:
                print(f"\n--- Sample {i+1}/{batch_size} ---")
            print(text)
        
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
            "generated_texts": generated_texts,
            "num_runs": num_runs,
            "batch_size": batch_size
        }
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print(f"Model device: {device}")
        if batch_size > 1:
            print(f"Input device: {input_ids.device}")
            print(f"Attention mask device: {attention_mask.device}")
        else:
            print(f"Input device: {input_ids.device}")
            
        # Return error information
        return {
            "kernel_mode": kernel_mode,
            "error": str(e),
            "device": str(device),
            "load_time": load_time
        }

def compare_kernels(num_tokens=100, temperature=0.7, prompt=None, num_runs=3, warmup_tokens=30, batch_size=1):
    """Compare stock vs hybrid kernels and print a summary."""
    results = []
    
    # Try to load optimizations before running
    if not AMD_SUPPORT:
        print("\n⚠️ WARNING: AMD optimizations module could not be imported.")
        print("Hybrid kernel benchmark may not show expected performance improvements.")
        print("Ensure 'mlstm_kernels.triton.amd_optimizations' is properly installed.\n")
    
    # Test stock kernels
    stock_results = run_benchmark("stock", num_tokens, temperature, prompt, num_runs, warmup_tokens, batch_size)
    results.append(stock_results)
    
    # Clear CUDA cache between runs and cool down period
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("\nCooling down for 3 seconds before next test...")
    time.sleep(3.0)
    
    # Test hybrid kernels
    hybrid_results = run_benchmark("hybrid", num_tokens, temperature, prompt, num_runs, warmup_tokens, batch_size)
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

def print_usage():
    """Print script usage information with examples."""
    print("\nxLSTM-7B - Text Generation and Benchmarking Tool")
    print("=" * 50)
    print("\nThis script allows you to run the xLSTM-7B model with different")
    print("kernel backends (stock, hybrid, or auto-detect) for text generation")
    print("and performance benchmarking.\n")
    
    print("USAGE EXAMPLES:")
    print("-" * 50)
    print("1. Simple text generation:")
    print("   python xlstm7b.py --mode generate --prompt \"Tell me a story about robots\"")
    print("\n2. Benchmark comparison between stock and hybrid kernels:")
    print("   python xlstm7b.py --mode compare")
    print("\n3. Benchmark specific kernel type:")
    print("   python xlstm7b.py --mode benchmark --kernel hybrid")
    print("\n4. Generate with specific parameters:")
    print("   python xlstm7b.py --mode generate --kernel hybrid --tokens 200 --temp 0.8")
    print("\n5. Batch generation:")
    print("   python xlstm7b.py --mode generate --batch-size 4 --prompt \"Write a poem about\"")
    print("-" * 50)
    
    print("\nAdditional options can be viewed with:")
    print("python xlstm7b.py --help")

def interactive_menu():
    """Display an interactive menu for the user to select options."""
    while True:
        print("\nxLSTM-7B - Text Generation and Benchmarking Tool")
        print("=" * 50)
        print("\nPlease select an option:")
        print("1. Generate text with xLSTM-7B")
        print("2. Benchmark a specific kernel (stock/hybrid/auto)")
        print("3. Compare stock vs hybrid kernels")
        print("4. Exit")
        
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == '1':  # Generate text
                kernel = input("\nSelect kernel (stock/hybrid/auto) [default: auto]: ").strip().lower() or 'auto'
                if kernel not in ['stock', 'hybrid', 'auto']:
                    print(f"Invalid kernel choice: {kernel}. Using 'auto' instead.")
                    kernel = 'auto'
                
                prompt = input("\nEnter your prompt [default: 'In a world where technology and nature coexist,']: ").strip()
                prompt = prompt or "In a world where technology and nature coexist,"
                
                tokens = input("\nNumber of tokens to generate [default: 100]: ").strip()
                tokens = int(tokens) if tokens.isdigit() else 100
                
                temp = input("\nTemperature (0.1-2.0) [default: 0.7]: ").strip()
                try:
                    temp = float(temp) if temp else 0.7
                    if temp < 0.1 or temp > 2.0:
                        print(f"Temperature {temp} out of recommended range (0.1-2.0), but proceeding anyway.")
                except ValueError:
                    print(f"Invalid temperature: {temp}. Using default 0.7.")
                    temp = 0.7
                
                batch_size = input("\nBatch size [default: 1]: ").strip()
                batch_size = int(batch_size) if batch_size.isdigit() and int(batch_size) > 0 else 1
                
                print(f"\nGenerating text with: kernel={kernel}, tokens={tokens}, temp={temp}, batch_size={batch_size}")
                print(f"Prompt: '{prompt}'")
                confirm = input("\nConfirm (y/n): ").strip().lower()
                if confirm == 'y':
                    simple_generation(kernel, tokens, temp, prompt, batch_size)
                else:
                    print("Generation cancelled.")
            
            elif choice == '2':  # Benchmark
                kernel = input("\nSelect kernel to benchmark (stock/hybrid/auto) [default: hybrid]: ").strip().lower() or 'hybrid'
                if kernel not in ['stock', 'hybrid', 'auto']:
                    print(f"Invalid kernel choice: {kernel}. Using 'hybrid' instead.")
                    kernel = 'hybrid'
                
                prompt = input("\nEnter your prompt [default: 'In a world where technology and nature coexist,']: ").strip()
                prompt = prompt or "In a world where technology and nature coexist,"
                
                tokens = input("\nNumber of tokens to generate [default: 100]: ").strip()
                tokens = int(tokens) if tokens.isdigit() else 100
                
                runs = input("\nNumber of benchmark runs [default: 3]: ").strip()
                runs = int(runs) if runs.isdigit() and int(runs) > 0 else 3
                
                warmup = input("\nWarmup tokens [default: 30]: ").strip()
                warmup = int(warmup) if warmup.isdigit() and int(warmup) > 0 else 30
                
                batch_size = input("\nBatch size [default: 1]: ").strip()
                batch_size = int(batch_size) if batch_size.isdigit() and int(batch_size) > 0 else 1
                
                force_amd = input("\nForce AMD detection (y/n) [default: y]: ").strip().lower() or 'y'
                if force_amd == 'y':
                    os.environ["FORCE_AMD_DETECTION"] = "1"
                    print("Forcing AMD detection for all runs")
                
                print(f"\nBenchmarking with: kernel={kernel}, tokens={tokens}, runs={runs}, warmup={warmup}, batch_size={batch_size}")
                print(f"Prompt: '{prompt}'")
                confirm = input("\nConfirm (y/n): ").strip().lower()
                if confirm == 'y':
                    run_benchmark(kernel, tokens, 0.7, prompt, runs, warmup, batch_size)
                else:
                    print("Benchmark cancelled.")
            
            elif choice == '3':  # Compare
                prompt = input("\nEnter your prompt [default: 'In a world where technology and nature coexist,']: ").strip()
                prompt = prompt or "In a world where technology and nature coexist,"
                
                tokens = input("\nNumber of tokens to generate [default: 100]: ").strip()
                tokens = int(tokens) if tokens.isdigit() else 100
                
                runs = input("\nNumber of benchmark runs [default: 3]: ").strip()
                runs = int(runs) if runs.isdigit() and int(runs) > 0 else 3
                
                warmup = input("\nWarmup tokens [default: 30]: ").strip()
                warmup = int(warmup) if warmup.isdigit() and int(warmup) > 0 else 30
                
                batch_size = input("\nBatch size [default: 1]: ").strip()
                batch_size = int(batch_size) if batch_size.isdigit() and int(batch_size) > 0 else 1
                
                force_amd = input("\nForce AMD detection (y/n) [default: y]: ").strip().lower() or 'y'
                if force_amd == 'y':
                    os.environ["FORCE_AMD_DETECTION"] = "1"
                    print("Forcing AMD detection for all runs")
                
                print(f"\nComparing kernels with: tokens={tokens}, runs={runs}, warmup={warmup}, batch_size={batch_size}")
                print(f"Prompt: '{prompt}'")
                confirm = input("\nConfirm (y/n): ").strip().lower()
                if confirm == 'y':
                    # Temporarily suspend keyboard interrupts during comparison
                    # to prevent interrupting between the stock and hybrid runs
                    try:
                        # Save the default handler
                        default_handler = signal.getsignal(signal.SIGINT)
                        # Set a temporary handler that ignores the signal
                        signal.signal(signal.SIGINT, lambda sig, frame: print('\nPlease wait for the comparison to complete...'))
                        
                        # Run the comparison
                        compare_kernels(tokens, 0.7, prompt, runs, warmup, batch_size)
                        
                        # Restore the default handler
                        signal.signal(signal.SIGINT, default_handler)
                    except Exception as e:
                        # Make sure we restore the handler even if there's an exception
                        signal.signal(signal.SIGINT, default_handler)
                        print(f"\nAn error occurred during comparison: {str(e)}")
                else:
                    print("Comparison cancelled.")
            
            elif choice == '4':  # Exit
                print("Exiting. Goodbye!")
                break
            
            else:
                print(f"Invalid choice: {choice}. Please enter 1, 2, 3, or 4.")
        
        except KeyboardInterrupt:
            print("\nOperation cancelled by user. Returning to menu.")
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")

def main():
    """Main entry point for the script."""
    # Create argument parser
    parser = argparse.ArgumentParser(
        description='xLSTM-7B text generation and benchmarking tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''
        Examples:
          python xlstm7b.py --mode generate --prompt "Tell me a story about robots"
          python xlstm7b.py --mode compare --tokens 200 --runs 5
          python xlstm7b.py --mode benchmark --kernel hybrid --batch-size 4
        ''')
    )
    
    # Add arguments
    parser.add_argument('--mode', type=str, default=None, 
                      choices=['generate', 'benchmark', 'compare', 'help'],
                      help='Operation mode: generate text, benchmark performance, compare kernels, or show help')
    
    parser.add_argument('--kernel', type=str, default='auto', 
                      choices=['stock', 'hybrid', 'auto'],
                      help='Kernel mode: stock (original), hybrid (AMD optimized), or auto (hardware detection)')
    
    parser.add_argument('--tokens', type=int, default=100, 
                      help='Number of tokens to generate')
    
    parser.add_argument('--temp', type=float, default=0.7, 
                      help='Temperature for generation (higher = more random)')
    
    parser.add_argument('--prompt', type=str, default=None, 
                      help='Text prompt for generation')
    
    parser.add_argument('--force-amd', action='store_true', 
                      help='Force AMD detection for all runs')
    
    parser.add_argument('--runs', type=int, default=3, 
                      help='Number of benchmark runs to average')
    
    parser.add_argument('--warmup-tokens', type=int, default=30, 
                      help='Number of tokens to generate in warmup (benchmarking only)')
    
    parser.add_argument('--batch-size', type=int, default=1, 
                      help='Batch size for generation')
    
    # Parse arguments
    args = parser.parse_args()
    
    # If no mode specified, launch interactive menu
    if args.mode is None and len(sys.argv) == 1:
        interactive_menu()
        return
    
    # If help mode or no arguments, show usage
    if args.mode == 'help':
        print_usage()
        return
    
    # Set global environment variable for AMD detection if requested
    if args.force_amd:
        os.environ["FORCE_AMD_DETECTION"] = "1"
        print("Forcing AMD detection for all runs")
    
    # Execute requested mode
    if args.mode == 'compare':
        compare_kernels(args.tokens, args.temp, args.prompt, args.runs, args.warmup_tokens, args.batch_size)
    
    elif args.mode == 'benchmark':
        run_benchmark(args.kernel, args.tokens, args.temp, args.prompt, args.runs, args.warmup_tokens, args.batch_size)
    
    elif args.mode == 'generate':
        simple_generation(args.kernel, args.tokens, args.temp, args.prompt, args.batch_size)
    
    else:
        print(f"Unknown mode: {args.mode}")
        print_usage()

if __name__ == "__main__":
    main() 