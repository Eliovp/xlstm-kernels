#!/usr/bin/env python3
# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import argparse
import os
import time
from typing import Dict, List, Optional, Tuple, Union

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mlstm_kernels.torch import get_mlstm_kernel

try:
    from mlstm_kernels.triton.amd_optimizations import is_amd, is_cdna3
    AMD_SUPPORT = True
except ImportError:
    AMD_SUPPORT = False
    
    def is_amd():
        return False
    
    def is_cdna3():
        return False

def run_benchmark(
    batch_size: int = 2,
    seq_len: int = 4096,
    head_dim_qk: int = 64,
    head_dim_v: int = 64,
    num_heads: int = 8,
    chunk_size: int = 128,
    dtype: torch.dtype = torch.float16,
    num_warmup: int = 5,
    num_repeats: int = 10,
    device: str = None,
):
    # Determine device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    # Generate input data
    q = torch.randn(
        (batch_size, num_heads, seq_len, head_dim_qk),
        dtype=dtype,
        device=device,
    )
    k = torch.randn(
        (batch_size, num_heads, seq_len, head_dim_qk),
        dtype=dtype,
        device=device,
    )
    v = torch.randn(
        (batch_size, num_heads, seq_len, head_dim_v),
        dtype=dtype,
        device=device,
    )
    
    # Create gate vectors
    vecI = torch.zeros(
        (batch_size, num_heads, seq_len),
        dtype=dtype,
        device=device,
    )
    vecF = torch.zeros(
        (batch_size, num_heads, seq_len),
        dtype=dtype,
        device=device,
    )
    
    # Get the xlstm kernel
    kernel = get_mlstm_kernel("chunkwise--triton_xl_chunk")
    
    # Benchmark forward pass
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    # Warmup
    for _ in range(num_warmup):
        out = kernel(q=q, k=k, v=v, i=vecI, f=vecF, return_last_states=False, chunk_size=chunk_size)
    
    # Measure
    torch.cuda.synchronize()
    times = []
    
    for _ in range(num_repeats):
        start.record()
        out = kernel(q=q, k=k, v=v, i=vecI, f=vecF, return_last_states=False, chunk_size=chunk_size)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    avg_time = sum(times) / len(times)
    std_time = np.std(times)
    
    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "head_dim_qk": head_dim_qk,
        "head_dim_v": head_dim_v,
        "num_heads": num_heads,
        "chunk_size": chunk_size,
        "dtype": str(dtype).split(".")[-1],
        "device": str(device),
        "is_amd": "Yes" if is_amd() else "No",
        "is_cdna3": "Yes" if AMD_SUPPORT and is_cdna3() else "No",
        "avg_time_ms": avg_time,
        "std_time_ms": std_time,
    }


def benchmark_model_sizes(
    batch_sizes: List[int] = [1, 2, 4, 8],
    seq_lens: List[int] = [2048, 4096, 8192],
    head_dims: List[int] = [64, 128],
    num_heads_list: List[int] = [8, 16],
    chunk_sizes: List[int] = [128, 256],
    dtypes: List[torch.dtype] = [torch.float16, torch.float32],
    num_warmup: int = 5,
    num_repeats: int = 10,
    device: str = None,
    output_file: str = "benchmark_results.csv",
):
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
                                  f"head_dim={head_dim}, heads={num_heads}, chunk={chunk_size}, "
                                  f"dtype={dtype}")
                            try:
                                result = run_benchmark(
                                    batch_size=batch_size,
                                    seq_len=seq_len,
                                    head_dim_qk=head_dim,
                                    head_dim_v=head_dim,
                                    num_heads=num_heads,
                                    chunk_size=chunk_size,
                                    dtype=dtype,
                                    num_warmup=num_warmup,
                                    num_repeats=num_repeats,
                                    device=device,
                                )
                                results.append(result)
                                # Save incrementally in case of crashes
                                df = pd.DataFrame(results)
                                df.to_csv(output_file, index=False)
                            except Exception as e:
                                print(f"Error benchmarking configuration: {e}")
    
    return results


def plot_results(results_file: str, output_dir: str = "benchmark_plots"):
    """Generate plots from benchmark results."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    df = pd.read_csv(results_file)
    
    # 1. Plot performance by sequence length
    plt.figure(figsize=(12, 8))
    for head_dim in df["head_dim_qk"].unique():
        subset = df[(df["head_dim_qk"] == head_dim) & 
                    (df["batch_size"] == df["batch_size"].min()) &
                    (df["dtype"] == "float16")]
        plt.plot(subset["seq_len"], subset["avg_time_ms"], 
                 marker='o', label=f"Head Dim: {head_dim}")
    
    plt.title("xLSTM Performance by Sequence Length")
    plt.xlabel("Sequence Length")
    plt.ylabel("Time (ms)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "performance_by_seq_len.png"), dpi=300)
    
    # 2. Compare AMD vs NVIDIA if both present in data
    if len(df["is_amd"].unique()) > 1:
        plt.figure(figsize=(12, 8))
        for is_amd in [True, False]:
            subset = df[(df["is_amd"] == ("Yes" if is_amd else "No")) & 
                        (df["dtype"] == "float16") &
                        (df["head_dim_qk"] == 64)]
            if not subset.empty:
                plt.plot(subset["seq_len"], subset["avg_time_ms"], 
                         marker='o', label="AMD" if is_amd else "NVIDIA")
        
        plt.title("AMD vs NVIDIA Performance Comparison")
        plt.xlabel("Sequence Length")
        plt.ylabel("Time (ms)")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "amd_vs_nvidia.png"), dpi=300)
    
    # 3. Heatmap of performance by batch size and sequence length
    heatmap_data = df[df["dtype"] == "float16"].pivot_table(
        values="avg_time_ms", 
        index="batch_size",
        columns="seq_len", 
        aggfunc="mean"
    )
    
    plt.figure(figsize=(12, 8))
    plt.imshow(heatmap_data, cmap="viridis", aspect="auto", interpolation="nearest")
    plt.colorbar(label="Time (ms)")
    plt.title("Performance Heatmap (Batch Size vs Sequence Length)")
    plt.xlabel("Sequence Length")
    plt.ylabel("Batch Size")
    plt.xticks(range(len(heatmap_data.columns)), heatmap_data.columns)
    plt.yticks(range(len(heatmap_data.index)), heatmap_data.index)
    
    # Add text annotations
    for i in range(len(heatmap_data.index)):
        for j in range(len(heatmap_data.columns)):
            plt.text(j, i, f"{heatmap_data.iloc[i, j]:.2f}", 
                     ha="center", va="center", color="white")
    
    plt.savefig(os.path.join(output_dir, "heatmap.png"), dpi=300)


def main():
    parser = argparse.ArgumentParser(description="Benchmark AMD vs NVIDIA xLSTM Performance")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4],
                        help="Batch sizes to benchmark")
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[2048, 4096, 8192],
                        help="Sequence lengths to benchmark")
    parser.add_argument("--head-dims", type=int, nargs="+", default=[64, 128],
                        help="Head dimensions to benchmark")
    parser.add_argument("--num-heads", type=int, nargs="+", default=[8, 16],
                        help="Number of heads to benchmark")
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[128, 256],
                        help="Chunk sizes to benchmark")
    parser.add_argument("--dtypes", type=str, nargs="+", default=["float16"],
                        choices=["float16", "float32"],
                        help="Data types to benchmark")
    parser.add_argument("--num-warmup", type=int, default=5,
                        help="Number of warmup iterations")
    parser.add_argument("--num-repeats", type=int, default=10,
                        help="Number of measurement iterations")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run on (default: auto-detect)")
    parser.add_argument("--output-file", type=str, default="benchmark_results.csv",
                        help="Output file for benchmark results")
    parser.add_argument("--plot", action="store_true",
                        help="Generate plots from benchmark results")
    parser.add_argument("--plot-only", action="store_true",
                        help="Only generate plots from existing results file")
    parser.add_argument("--output-dir", type=str, default="benchmark_plots",
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
        plot_results(args.output_file, args.output_dir)
        return
    
    results = benchmark_model_sizes(
        batch_sizes=args.batch_sizes,
        seq_lens=args.seq_lens,
        head_dims=args.head_dims,
        num_heads_list=args.num_heads,
        chunk_sizes=args.chunk_sizes,
        dtypes=dtypes,
        num_warmup=args.num_warmup,
        num_repeats=args.num_repeats,
        device=args.device,
        output_file=args.output_file,
    )
    
    df = pd.DataFrame(results)
    df.to_csv(args.output_file, index=False)
    print(f"Benchmark results saved to {args.output_file}")
    
    if args.plot:
        plot_results(args.output_file, args.output_dir)
        print(f"Plots saved to directory: {args.output_dir}")


if __name__ == "__main__":
    main() 