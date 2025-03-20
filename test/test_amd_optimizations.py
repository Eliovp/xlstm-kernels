#!/usr/bin/env python3
# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import torch
import time
import argparse
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

# Fix the import to use the correct path
from mlstm_kernels.torch import get_mlstm_kernel

try:
    from mlstm_kernels.triton.amd_optimizations import is_amd, get_amd_warp_size, is_cdna3
    AMD_SUPPORT = True
except ImportError:
    AMD_SUPPORT = False
    
    def is_amd():
        return False
    
    def get_amd_warp_size():
        return 64
    
    def is_cdna3():
        return False


def test_amd_optimizations():
    """Test AMD optimizations and print hardware/environment info."""
    # Print hardware and environment information
    print("\n=== AMD Optimization Testing ===\n")
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current device: {torch.cuda.get_device_name()}")
        print(f"Device count: {torch.cuda.device_count()}")
    
    # Check if running on AMD GPU
    print(f"\nRunning on AMD GPU: {is_amd()}")
    if AMD_SUPPORT:
        print(f"AMD wavefront size: {get_amd_warp_size()}")
        print(f"Is CDNA3 architecture: {is_cdna3()}")
    
    if not torch.cuda.is_available():
        print("\nNo GPU available. Skipping kernel tests.")
        return
    
    # Basic kernel execution test
    print("\n=== Testing xLSTM Kernel Performance ===\n")
    
    # Setup parameters for a basic test
    batch_size = 1
    seq_len = 2048
    head_dim = 64
    num_heads = 8
    chunk_size = 128
    dtype = torch.float16
    iterations = 10
    
    # Create tensors
    q = torch.randn(
        (batch_size, num_heads, seq_len, head_dim),
        dtype=dtype,
        device="cuda",
    )
    k = torch.randn(
        (batch_size, num_heads, seq_len, head_dim),
        dtype=dtype,
        device="cuda",
    )
    v = torch.randn(
        (batch_size, num_heads, seq_len, head_dim),
        dtype=dtype,
        device="cuda",
    )
    
    # Create gate vectors with correct shapes (B, NH, S)
    vecI = torch.zeros(
        (batch_size, num_heads, seq_len),
        dtype=dtype,
        device="cuda",
    )
    vecF = torch.zeros(
        (batch_size, num_heads, seq_len),
        dtype=dtype,
        device="cuda",
    )
    
    # Get the kernel
    kernel = get_mlstm_kernel("chunkwise--triton_xl_chunk")
    
    # Warmup
    for _ in range(3):
        out = kernel(q=q, k=k, v=v, i=vecI, f=vecF, return_last_states=False, chunk_size=chunk_size)
    
    # Measure performance
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(iterations):
        out = kernel(q=q, k=k, v=v, i=vecI, f=vecF, return_last_states=False, chunk_size=chunk_size)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time_ms = (end_time - start_time) * 1000 / iterations
    
    print(f"Average execution time: {avg_time_ms:.2f} ms")
    print(f"Sequence length: {seq_len}, Head dim: {head_dim}, Heads: {num_heads}")
    
    # Test correctness by checking for NaNs
    if torch.isnan(out).any():
        print("WARNING: Output contains NaN values")
    else:
        print("Output verification: No NaN values found")
    
    print("\n=== Test Complete ===\n")


def test_batch_sequence_scaling():
    """Test how performance scales with batch size and sequence length."""
    if not torch.cuda.is_available():
        print("No GPU available. Skipping scaling tests.")
        return
    
    print("\n=== Testing Scaling Properties ===\n")
    
    # Fixed parameters
    head_dim = 64
    num_heads = 8
    chunk_size = 128
    dtype = torch.float16
    iterations = 5
    
    # Test different batch sizes and sequence lengths
    batch_sizes = [1, 2, 4]
    seq_lens = [1024, 2048, 4096]
    
    results = []
    
    # Get the kernel
    kernel = get_mlstm_kernel("chunkwise--triton_xl_chunk")
    
    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            # Skip configurations that would be too large for memory
            if batch_size * seq_len > 16384:
                continue
                
            # Create tensors
            q = torch.randn(
                (batch_size, num_heads, seq_len, head_dim),
                dtype=dtype,
                device="cuda",
            )
            k = torch.randn(
                (batch_size, num_heads, seq_len, head_dim),
                dtype=dtype,
                device="cuda",
            )
            v = torch.randn(
                (batch_size, num_heads, seq_len, head_dim),
                dtype=dtype,
                device="cuda",
            )
            
            # Create gate vectors with correct shapes (B, NH, S)
            vecI = torch.zeros(
                (batch_size, num_heads, seq_len),
                dtype=dtype,
                device="cuda",
            )
            vecF = torch.zeros(
                (batch_size, num_heads, seq_len),
                dtype=dtype,
                device="cuda",
            )
            
            # Warmup
            for _ in range(2):
                out = kernel(q=q, k=k, v=v, i=vecI, f=vecF, return_last_states=False, chunk_size=chunk_size)
            
            # Measure performance
            torch.cuda.synchronize()
            start_time = time.time()
            
            for _ in range(iterations):
                out = kernel(q=q, k=k, v=v, i=vecI, f=vecF, return_last_states=False, chunk_size=chunk_size)
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            avg_time_ms = (end_time - start_time) * 1000 / iterations
            
            print(f"Batch: {batch_size}, Seq: {seq_len} - Time: {avg_time_ms:.2f} ms")
            
            results.append({
                "batch_size": batch_size,
                "seq_len": seq_len,
                "time_ms": avg_time_ms
            })
    
    # Print scaling analysis
    if results:
        print("\n=== Scaling Analysis ===\n")
        
        # Analyze batch size scaling (keeping seq_len constant)
        base_seq_len = seq_lens[0]
        batch_scaling = []
        
        for i in range(len(batch_sizes) - 1):
            for result in results:
                if result["seq_len"] == base_seq_len and result["batch_size"] == batch_sizes[i]:
                    time1 = result["time_ms"]
                    
            for result in results:
                if result["seq_len"] == base_seq_len and result["batch_size"] == batch_sizes[i + 1]:
                    time2 = result["time_ms"]
            
            if "time1" in locals() and "time2" in locals():
                batch_ratio = batch_sizes[i + 1] / batch_sizes[i]
                time_ratio = time2 / time1
                batch_scaling.append(time_ratio / batch_ratio)
        
        if batch_scaling:
            avg_batch_scaling = sum(batch_scaling) / len(batch_scaling)
            print(f"Batch size scaling factor: {avg_batch_scaling:.2f}x (ideal: 1.0x)")
        
        # Analyze sequence length scaling (keeping batch_size constant)
        base_batch_size = batch_sizes[0]
        seq_scaling = []
        
        for i in range(len(seq_lens) - 1):
            for result in results:
                if result["batch_size"] == base_batch_size and result["seq_len"] == seq_lens[i]:
                    time1 = result["time_ms"]
                    
            for result in results:
                if result["batch_size"] == base_batch_size and result["seq_len"] == seq_lens[i + 1]:
                    time2 = result["time_ms"]
            
            if "time1" in locals() and "time2" in locals():
                seq_ratio = seq_lens[i + 1] / seq_lens[i]
                time_ratio = time2 / time1
                seq_scaling.append(time_ratio / seq_ratio)
        
        if seq_scaling:
            avg_seq_scaling = sum(seq_scaling) / len(seq_scaling)
            print(f"Sequence length scaling factor: {avg_seq_scaling:.2f}x (ideal: 1.0x)")


def main():
    parser = argparse.ArgumentParser(description="Test AMD optimizations for xLSTM kernels")
    parser.add_argument("--basic", action="store_true", 
                        help="Run basic optimization test")
    parser.add_argument("--scaling", action="store_true", 
                        help="Run scaling test with different batch sizes and sequence lengths")
    parser.add_argument("--all", action="store_true", 
                        help="Run all tests")
    
    args = parser.parse_args()
    
    # Default to basic test if no args specified
    if not (args.basic or args.scaling or args.all):
        args.basic = True
    
    if args.basic or args.all:
        test_amd_optimizations()
    
    if args.scaling or args.all:
        test_batch_sequence_scaling()


if __name__ == "__main__":
    main() 