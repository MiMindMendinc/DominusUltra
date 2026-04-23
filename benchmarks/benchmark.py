"""
Performance benchmarking for Dominus Ultra attention kernels.

Compares Dominus Ultra against PyTorch's native scaled_dot_product_attention
across various configurations, measuring throughput and memory usage.
"""

import torch
import torch.nn.functional as F
import time
from typing import Dict, List, Tuple
import argparse

from dominus_ultra import (
    dominus_ultra_prefill,
    dominus_ultra_decode,
    precompute_rope_cos_sin,
)


def benchmark_function(func, *args, warmup=5, iterations=100):
    """
    Benchmark a function's execution time.
    
    Args:
        func: Function to benchmark
        *args: Arguments to pass to function
        warmup: Number of warmup iterations
        iterations: Number of timed iterations
    
    Returns:
        Tuple of (mean_time_ms, std_time_ms, throughput_tokens_per_sec)
    """
    # Warmup
    for _ in range(warmup):
        func(*args)
    
    torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = func(*args)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    mean_time = sum(times) / len(times)
    std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5
    
    return mean_time, std_time


def apply_rope_reference(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Reference RoPE implementation for benchmarking."""
    dim = x.shape[-1]
    half = dim // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def benchmark_prefill(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    num_kv_heads: int = None,
    dtype=torch.bfloat16,
    device="cuda",
):
    """
    Benchmark prefill kernel against PyTorch reference.
    
    Args:
        batch_size: Batch size
        num_heads: Number of query heads
        seq_len: Sequence length
        head_dim: Head dimension
        num_kv_heads: Number of KV heads (for GQA). If None, uses num_heads (MHA)
        dtype: Data type
        device: Device
    
    Returns:
        Dictionary with benchmark results
    """
    if num_kv_heads is None:
        num_kv_heads = num_heads
    
    # Create inputs
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device, dtype=dtype)
    
    cos, sin = precompute_rope_cos_sin(seq_len, head_dim, device, dtype)
    
    # Benchmark Dominus Ultra
    def run_triton():
        return dominus_ultra_prefill(q, k, v, cos, sin, num_kv_heads=num_kv_heads)
    
    triton_time, triton_std = benchmark_function(run_triton)
    
    # Benchmark PyTorch reference
    group_size = num_heads // num_kv_heads
    k_expanded = k.repeat_interleave(group_size, dim=1)
    v_expanded = v.repeat_interleave(group_size, dim=1)
    
    q_rope = apply_rope_reference(q, cos, sin)
    k_rope = apply_rope_reference(k_expanded, cos, sin)
    
    def run_pytorch():
        return F.scaled_dot_product_attention(q_rope, k_rope, v_expanded, is_causal=True)
    
    pytorch_time, pytorch_std = benchmark_function(run_pytorch)
    
    # Calculate tokens per second
    total_tokens = batch_size * seq_len
    triton_throughput = total_tokens / (triton_time / 1000)
    pytorch_throughput = total_tokens / (pytorch_time / 1000)
    
    speedup = pytorch_time / triton_time
    
    return {
        "config": f"B={batch_size}, H={num_heads}, KV_H={num_kv_heads}, T={seq_len}, D={head_dim}",
        "triton_time_ms": triton_time,
        "triton_std_ms": triton_std,
        "pytorch_time_ms": pytorch_time,
        "pytorch_std_ms": pytorch_std,
        "speedup": speedup,
        "triton_tokens_per_sec": triton_throughput,
        "pytorch_tokens_per_sec": pytorch_throughput,
    }


def benchmark_decode(
    batch_size: int,
    num_heads: int,
    past_len: int,
    head_dim: int,
    num_kv_heads: int = None,
    dtype=torch.bfloat16,
    device="cuda",
):
    """
    Benchmark decode kernel against PyTorch reference.
    
    Args:
        batch_size: Batch size
        num_heads: Number of query heads
        past_len: Length of KV cache
        head_dim: Head dimension
        num_kv_heads: Number of KV heads (for GQA). If None, uses num_heads (MHA)
        dtype: Data type
        device: Device
    
    Returns:
        Dictionary with benchmark results
    """
    if num_kv_heads is None:
        num_kv_heads = num_heads
    
    # Create inputs
    q_new = torch.randn(batch_size, num_heads, 1, head_dim, device=device, dtype=dtype)
    k_cache = torch.randn(batch_size, num_kv_heads, past_len, head_dim, device=device, dtype=dtype)
    v_cache = torch.randn(batch_size, num_kv_heads, past_len, head_dim, device=device, dtype=dtype)
    
    cos, sin = precompute_rope_cos_sin(past_len + 1, head_dim, device, dtype)
    
    # Benchmark Dominus Ultra
    def run_triton():
        return dominus_ultra_decode(q_new, k_cache, v_cache, cos, sin, num_kv_heads=num_kv_heads)
    
    triton_time, triton_std = benchmark_function(run_triton)
    
    # Benchmark PyTorch reference
    group_size = num_heads // num_kv_heads
    k_expanded = k_cache.repeat_interleave(group_size, dim=1)
    v_expanded = v_cache.repeat_interleave(group_size, dim=1)
    
    q_rope = apply_rope_reference(q_new, cos[past_len:past_len+1], sin[past_len:past_len+1])
    k_rope = apply_rope_reference(k_expanded, cos[:past_len], sin[:past_len])
    
    def run_pytorch():
        return F.scaled_dot_product_attention(q_rope, k_rope, v_expanded, is_causal=False)
    
    pytorch_time, pytorch_std = benchmark_function(run_pytorch)
    
    # Calculate tokens per second
    total_tokens = batch_size
    triton_throughput = total_tokens / (triton_time / 1000)
    pytorch_throughput = total_tokens / (pytorch_time / 1000)
    
    speedup = pytorch_time / triton_time
    
    return {
        "config": f"B={batch_size}, H={num_heads}, KV_H={num_kv_heads}, past_T={past_len}, D={head_dim}",
        "triton_time_ms": triton_time,
        "triton_std_ms": triton_std,
        "pytorch_time_ms": pytorch_time,
        "pytorch_std_ms": pytorch_std,
        "speedup": speedup,
        "triton_tokens_per_sec": triton_throughput,
        "pytorch_tokens_per_sec": pytorch_throughput,
    }


def print_results(results: List[Dict], title: str):
    """Pretty print benchmark results."""
    print(f"\n{'=' * 100}")
    print(f"{title:^100}")
    print(f"{'=' * 100}")
    print(f"{'Config':<50} {'Triton (ms)':<15} {'PyTorch (ms)':<15} {'Speedup':<10}")
    print(f"{'-' * 100}")
    
    for r in results:
        print(f"{r['config']:<50} "
              f"{r['triton_time_ms']:>6.3f}±{r['triton_std_ms']:>5.3f}   "
              f"{r['pytorch_time_ms']:>6.3f}±{r['pytorch_std_ms']:>5.3f}   "
              f"{r['speedup']:>6.2f}x")
    
    print(f"{'=' * 100}\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Dominus Ultra kernels")
    parser.add_argument("--mode", choices=["prefill", "decode", "all"], default="all",
                        help="Benchmark mode")
    parser.add_argument("--device", default="cuda", help="Device to run on")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"],
                        help="Data type")
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("CUDA not available, exiting.")
        return
    
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    device = args.device
    
    print(f"Running benchmarks on {device} with {args.dtype}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    
    if args.mode in ["prefill", "all"]:
        print("\n🚀 Benchmarking PREFILL kernels...")
        
        prefill_results = []
        
        # Multi-Head Attention configurations
        print("\n📊 Multi-Head Attention (MHA)")
        for seq_len in [128, 256, 512, 1024, 2048]:
            result = benchmark_prefill(
                batch_size=2,
                num_heads=32,
                seq_len=seq_len,
                head_dim=64,
                dtype=dtype,
                device=device,
            )
            prefill_results.append(result)
        
        print_results(prefill_results, "Prefill - Multi-Head Attention")
        
        # Grouped Query Attention configurations
        print("\n📊 Grouped Query Attention (GQA)")
        gqa_results = []
        for num_kv_heads in [8, 4, 2, 1]:
            result = benchmark_prefill(
                batch_size=2,
                num_heads=32,
                seq_len=1024,
                head_dim=64,
                num_kv_heads=num_kv_heads,
                dtype=dtype,
                device=device,
            )
            gqa_results.append(result)
        
        print_results(gqa_results, "Prefill - Grouped Query Attention")
    
    if args.mode in ["decode", "all"]:
        print("\n🚀 Benchmarking DECODE kernels...")
        
        decode_results = []
        
        # Multi-Head Attention decode
        print("\n📊 Decode with various cache sizes")
        for past_len in [128, 256, 512, 1024, 2048]:
            result = benchmark_decode(
                batch_size=8,
                num_heads=32,
                past_len=past_len,
                head_dim=64,
                dtype=dtype,
                device=device,
            )
            decode_results.append(result)
        
        print_results(decode_results, "Decode - Multi-Head Attention")
        
        # GQA decode
        print("\n📊 Decode with Grouped Query Attention")
        gqa_decode_results = []
        for num_kv_heads in [8, 4, 2, 1]:
            result = benchmark_decode(
                batch_size=8,
                num_heads=32,
                past_len=1024,
                head_dim=64,
                num_kv_heads=num_kv_heads,
                dtype=dtype,
                device=device,
            )
            gqa_decode_results.append(result)
        
        print_results(gqa_decode_results, "Decode - Grouped Query Attention")
    
    print("✨ Benchmarking complete!")


if __name__ == "__main__":
    main()
