#!/usr/bin/env python3
# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import torch
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import importlib
from typing import Dict, List, Optional, Tuple, Union

from mlstm_kernels.torch import get_mlstm_kernel

# Create a modified version of the AMD detection function that can be manually overridden
_FORCE_DISABLE_AMD = False

try:
    from mlstm_kernels.triton.amd_optimizations import is_amd as _original_is_amd
    from mlstm_kernels.triton.amd_optimizations import get_amd_warp_size, is_cdna3
    
    def is_amd():
        if _FORCE_DISABLE_AMD:
            return False
        return _original_is_amd()
    
    AMD_SUPPORT = True
except ImportError:
    AMD_SUPPORT = False
    
    def is_amd():
        return False
    
    def get_amd_warp_size():
        return 64
    
    def is_cdna3():
        return False


def patch_is_amd_function(enable_amd: bool):
    """Patch the is_amd function in kernel_param_heuristics."""
    import mlstm_kernels.triton.kernel_param_heuristics as kph
    
    # Set the override function
    def is_amd_override():
        return enable_amd
    
    kph.is_amd_override = is_amd_override


def run_kernel_benchmark(
    batch_size: int = 1,
    seq_len: int = 2048,
    head_dim: int = 64,
    num_heads: int = 8,
    chunk_size: int = 128,
    dtype: torch.dtype = torch.float16,
    num_warmup: int = 3,
    num_repeats: int = 10,
    use_amd_optimizations: bool = True,
):
    """Run a benchmark with or without AMD optimizations."""
    global _FORCE_DISABLE_AMD
    
    # Set AMD optimization flag
    _FORCE_DISABLE_AMD = not use_amd_optimizations
    
    # Patch the is_amd function in kernel_param_heuristics
    patch_is_amd_function(use_amd_optimizations)
    
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
    for _ in range(num_warmup):
        out = kernel(q=q, k=k, v=v, i=vecI, f=vecF, return_last_states=False, chunk_size=chunk_size)
    
    # Measure performance
    torch.cuda.synchronize()
    times = []
    
    for _ in range(num_repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        out = kernel(q=q, k=k, v=v, i=vecI, f=vecF, return_last_states=False, chunk_size=chunk_size)
        end.record()
        
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    # Print which implementation was used
    opt_status = "AMD-optimized" if use_amd_optimizations else "Stock"
    
    avg_time = sum(times) / len(times)
    std_time = np.std(times)
    
    print(f"  {opt_status}: avg_time={avg_time:.3f}ms, std={std_time:.3f}ms")
    
    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "head_dim": head_dim,
        "num_heads": num_heads,
        "chunk_size": chunk_size,
        "dtype": str(dtype).split(".")[-1],
        "optimized": use_amd_optimizations,
        "avg_time_ms": avg_time,
        "std_time_ms": std_time,
    }


def compare_kernels(
    batch_sizes: List[int] = [1, 2, 4],
    seq_lens: List[int] = [1024, 2048, 4096],
    head_dims: List[int] = [64, 128],
    num_heads_list: List[int] = [8],
    chunk_sizes: List[int] = [128],
    dtypes: List[torch.dtype] = [torch.float16],
    num_warmup: int = 3,
    num_repeats: int = 10,
    output_file: str = "kernel_comparison.csv",
):
    """Run comparisons between stock and AMD-optimized kernels."""
    results = []
    
    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            for head_dim in head_dims:
                for num_heads in num_heads_list:
                    for chunk_size in chunk_sizes:
                        if chunk_size > seq_len:
                            continue
                        for dtype in dtypes:
                            print(f"Benchmarking: batch={batch_size}, seq_len={seq_len}, "
                                  f"head_dim={head_dim}, heads={num_heads}, chunk={chunk_size}")
                            
                            # Run with stock kernels first
                            try:
                                stock_result = run_kernel_benchmark(
                                    batch_size=batch_size,
                                    seq_len=seq_len,
                                    head_dim=head_dim,
                                    num_heads=num_heads,
                                    chunk_size=chunk_size,
                                    dtype=dtype,
                                    num_warmup=num_warmup,
                                    num_repeats=num_repeats,
                                    use_amd_optimizations=False,
                                )
                                results.append(stock_result)
                            except Exception as e:
                                print(f"Error benchmarking stock configuration: {e}")
                                continue
                                
                            # Then run with AMD optimizations
                            try:
                                amd_result = run_kernel_benchmark(
                                    batch_size=batch_size,
                                    seq_len=seq_len,
                                    head_dim=head_dim,
                                    num_heads=num_heads,
                                    chunk_size=chunk_size,
                                    dtype=dtype,
                                    num_warmup=num_warmup,
                                    num_repeats=num_repeats,
                                    use_amd_optimizations=True,
                                )
                                results.append(amd_result)
                            except Exception as e:
                                print(f"Error benchmarking AMD-optimized configuration: {e}")
                                
                            # Calculate speedup
                            if len(results) >= 2 and results[-2]["optimized"] == False and results[-1]["optimized"] == True:
                                speedup = results[-2]["avg_time_ms"] / results[-1]["avg_time_ms"]
                                print(f"  Speedup: {speedup:.2f}x\n")
                                
                            # Save incrementally in case of crashes
                            df = pd.DataFrame(results)
                            df.to_csv(output_file, index=False)
    
    return results


def plot_comparison(results_file: str, output_dir: str = "comparison_plots"):
    """Generate comparison plots."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    df = pd.read_csv(results_file)
    
    # Add speedup column
    df_stock = df[df["optimized"] == False].copy()
    df_amd = df[df["optimized"] == True].copy()
    
    # Ensure we have matching configurations
    merged = pd.merge(
        df_stock, df_amd, 
        on=["batch_size", "seq_len", "head_dim", "num_heads", "chunk_size", "dtype"],
        suffixes=("_stock", "_amd")
    )
    
    merged["speedup"] = merged["avg_time_ms_stock"] / merged["avg_time_ms_amd"]
    
    # 1. Speedup by sequence length
    plt.figure(figsize=(12, 8))
    for head_dim in merged["head_dim"].unique():
        subset = merged[merged["head_dim"] == head_dim]
        if not subset.empty:
            plt.plot(subset["seq_len"], subset["speedup"], 
                    marker='o', label=f"Head Dim: {head_dim}")
    
    plt.axhline(y=1.0, color='r', linestyle='--')
    plt.title("AMD Optimization Speedup by Sequence Length")
    plt.xlabel("Sequence Length")
    plt.ylabel("Speedup (Stock/AMD)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "speedup_by_seq_len.png"), dpi=300)
    
    # 2. Speedup by batch size
    plt.figure(figsize=(12, 8))
    for seq_len in merged["seq_len"].unique():
        subset = merged[merged["seq_len"] == seq_len]
        if not subset.empty:
            plt.plot(subset["batch_size"], subset["speedup"], 
                    marker='o', label=f"Seq Len: {seq_len}")
    
    plt.axhline(y=1.0, color='r', linestyle='--')
    plt.title("AMD Optimization Speedup by Batch Size")
    plt.xlabel("Batch Size")
    plt.ylabel("Speedup (Stock/AMD)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "speedup_by_batch.png"), dpi=300)
    
    # 3. Bar chart of stock vs AMD
    plt.figure(figsize=(15, 10))
    
    # Create configuration labels
    merged["config"] = merged.apply(
        lambda row: f"B{row['batch_size']}_S{row['seq_len']}_H{row['head_dim']}", axis=1
    )
    
    configs = merged["config"].unique()
    stock_times = merged["avg_time_ms_stock"].values
    amd_times = merged["avg_time_ms_amd"].values
    
    x = np.arange(len(configs))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(15, 8))
    rects1 = ax.bar(x - width/2, stock_times, width, label='Stock')
    rects2 = ax.bar(x + width/2, amd_times, width, label='AMD Optimized')
    
    ax.set_ylabel('Time (ms)')
    ax.set_title('Stock vs AMD Optimized Kernel Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.legend()
    
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, "stock_vs_amd.png"), dpi=300)
    
    # 4. Summary table
    summary = pd.DataFrame({
        'Configuration': configs,
        'Stock (ms)': stock_times,
        'AMD Optimized (ms)': amd_times,
        'Speedup': merged["speedup"].values
    })
    
    summary.to_csv(os.path.join(output_dir, "summary.csv"), index=False)
    
    print("\n=== Performance Summary ===")
    print(f"Average Speedup: {merged['speedup'].mean():.2f}x")
    print(f"Max Speedup: {merged['speedup'].max():.2f}x")
    print(f"Min Speedup: {merged['speedup'].min():.2f}x")
    
    return merged


def main():
    parser = argparse.ArgumentParser(description="Compare AMD-optimized kernels with stock kernels")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4],
                        help="Batch sizes to benchmark")
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[2048, 4096, 8192],
                        help="Sequence lengths to benchmark")
    parser.add_argument("--head-dims", type=int, nargs="+", default=[64, 128],
                        help="Head dimensions to benchmark")
    parser.add_argument("--num-heads", type=int, nargs="+", default=[8],
                        help="Number of heads to benchmark")
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[128],
                        help="Chunk sizes to benchmark")
    parser.add_argument("--dtypes", type=str, nargs="+", default=["float16"],
                        choices=["float16", "float32"],
                        help="Data types to benchmark")
    parser.add_argument("--num-warmup", type=int, default=3,
                        help="Number of warmup iterations")
    parser.add_argument("--num-repeats", type=int, default=10,
                        help="Number of measurement iterations")
    parser.add_argument("--output-file", type=str, default="kernel_comparison.csv",
                        help="Output file for benchmark results")
    parser.add_argument("--plot", action="store_true",
                        help="Generate plots from benchmark results")
    parser.add_argument("--plot-only", action="store_true",
                        help="Only generate plots from existing results file")
    parser.add_argument("--output-dir", type=str, default="comparison_plots",
                        help="Directory to save plots")
    
    args = parser.parse_args()
    
    # Convert string dtype arguments to torch dtypes
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtypes = [dtype_map[d] for d in args.dtypes]
    
    if args.plot_only:
        if not os.path.exists(args.output_file):
            print(f"Error: Results file {args.output_file} not found.")
            return
        plot_comparison(args.output_file, args.output_dir)
        return
    
    results = compare_kernels(
        batch_sizes=args.batch_sizes,
        seq_lens=args.seq_lens,
        head_dims=args.head_dims,
        num_heads_list=args.num_heads,
        chunk_sizes=args.chunk_sizes,
        dtypes=dtypes,
        num_warmup=args.num_warmup,
        num_repeats=args.num_repeats,
        output_file=args.output_file,
    )
    
    df = pd.DataFrame(results)
    df.to_csv(args.output_file, index=False)
    print(f"Comparison results saved to {args.output_file}")
    
    if args.plot:
        plot_comparison(args.output_file, args.output_dir)
        print(f"Plots saved to directory: {args.output_dir}")


if __name__ == "__main__":
    main() 