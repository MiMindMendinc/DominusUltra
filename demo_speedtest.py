"""Recordable live benchmark demo for DominusUltra.

This script presents a terminal "speed test" for attention kernels:
DominusUltra versus a PyTorch reference path, with live progress bars and a
saved Markdown report for sharing after a screen recording.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

try:
    import triton  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - optional metadata only
    triton = None

from dominus_ultra import dominus_ultra_prefill, precompute_rope_cos_sin


RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
RED = "\033[31m"


def c(text: str, color: str) -> str:
    return f"{color}{text}{RESET}"


def apply_rope_reference(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    dim = x.shape[-1]
    half = dim // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def bar(label: str, index: int, total: int, width: int = 34) -> None:
    filled = int(width * index / max(total, 1))
    visual = "#" * filled + "-" * (width - filled)
    pct = int(100 * index / max(total, 1))
    sys.stdout.write(f"\r{label:<18} [{visual}] {pct:>3}%")
    sys.stdout.flush()
    if index >= total:
        sys.stdout.write("\n")


def time_call(fn: Callable[[], object], iterations: int, label: str) -> tuple[float, float]:
    times_ms: list[float] = []
    for i in range(1, iterations + 1):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - start) * 1000)
        bar(label, i, iterations)
    return statistics.mean(times_ms), statistics.pstdev(times_ms)


def countdown(seconds: int) -> None:
    for remaining in range(seconds, 0, -1):
        print(c(f"Starting live benchmark in {remaining}...", YELLOW))
        time.sleep(1)


def format_rate(tokens_per_second: float) -> str:
    if tokens_per_second >= 1_000_000:
        return f"{tokens_per_second / 1_000_000:.2f}M tokens/sec"
    if tokens_per_second >= 1_000:
        return f"{tokens_per_second / 1_000:.2f}K tokens/sec"
    return f"{tokens_per_second:.2f} tokens/sec"


def write_report(path: Path, result: dict) -> None:
    markdown = f"""# DominusUltra Live Speed Test

Generated: {result["generated_at"]}

## Hardware

- Device: {result["device_name"]}
- CUDA: {result["cuda_version"]}
- Python: {result["python_version"]}
- PyTorch: {result["torch_version"]}
- Triton: {result["triton_version"]}

## Configuration

- Batch size: {result["batch_size"]}
- Query heads: {result["num_heads"]}
- KV heads: {result["num_kv_heads"]}
- Sequence length: {result["seq_len"]}
- Head dimension: {result["head_dim"]}
- Dtype: {result["dtype"]}
- Warmup iterations: {result["warmup"]}
- Timed iterations: {result["iterations"]}

## Result

| Kernel | Latency | Throughput |
| --- | ---: | ---: |
| DominusUltra | {result["dominus_ms"]:.3f} ms +/- {result["dominus_std_ms"]:.3f} | {format_rate(result["dominus_tokens_per_sec"])} |
| PyTorch reference | {result["pytorch_ms"]:.3f} ms +/- {result["pytorch_std_ms"]:.3f} | {format_rate(result["pytorch_tokens_per_sec"])} |

Speedup: **{result["speedup"]:.2f}x**

Correctness max error: `{result["max_error"]:.6f}`
"""
    path.write_text(markdown, encoding="utf-8")
    path.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Live DominusUltra attention speed test")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--countdown", type=int, default=3)
    parser.add_argument("--output", default="benchmark_results/live_speedtest.md")
    args = parser.parse_args()

    print(c("\nDominusUltra Live Attention Speed Test", BOLD + CYAN))
    print(c("A recordable benchmark: fused Triton RoPE + GQA versus PyTorch reference.\n", DIM))

    if not torch.cuda.is_available():
        print(c("CUDA is required for the live Triton speed test.", RED))
        print("Run this on an NVIDIA GPU machine before recording the client demo.")
        return 2

    if args.num_heads % args.num_kv_heads != 0:
        print(c("num-heads must be divisible by num-kv-heads for GQA.", RED))
        return 2

    torch.manual_seed(7)
    device = "cuda"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    device_name = torch.cuda.get_device_name(0)
    triton_version = getattr(triton, "__version__", "unknown") if triton else "unknown"

    print(f"Device:       {device_name}")
    print(f"CUDA:         {torch.version.cuda}")
    print(f"PyTorch:      {torch.__version__}")
    print(f"Triton:       {triton_version}")
    print(f"Shape:        B={args.batch_size}, Hq={args.num_heads}, Hkv={args.num_kv_heads}, T={args.seq_len}, D={args.head_dim}")
    print(f"Dtype:        {args.dtype}")
    print()

    countdown(args.countdown)

    q = torch.randn(args.batch_size, args.num_heads, args.seq_len, args.head_dim, device=device, dtype=dtype)
    k = torch.randn(args.batch_size, args.num_kv_heads, args.seq_len, args.head_dim, device=device, dtype=dtype)
    v = torch.randn(args.batch_size, args.num_kv_heads, args.seq_len, args.head_dim, device=device, dtype=dtype)
    cos, sin = precompute_rope_cos_sin(args.seq_len, args.head_dim, device=device, dtype=dtype)

    group_size = args.num_heads // args.num_kv_heads
    k_expanded = k.repeat_interleave(group_size, dim=1)
    v_expanded = v.repeat_interleave(group_size, dim=1)
    q_rope = apply_rope_reference(q, cos, sin)
    k_rope = apply_rope_reference(k_expanded, cos, sin)

    def run_dominus() -> tuple[torch.Tensor, torch.Tensor]:
        return dominus_ultra_prefill(q, k, v, cos, sin, num_kv_heads=args.num_kv_heads)

    def run_pytorch() -> torch.Tensor:
        return F.scaled_dot_product_attention(q_rope, k_rope, v_expanded, is_causal=True)

    print(c("\nWarmup", BOLD))
    for i in range(1, args.warmup + 1):
        run_dominus()
        run_pytorch()
        torch.cuda.synchronize()
        bar("warming kernels", i, args.warmup)

    print(c("\nLive run", BOLD))
    dominus_ms, dominus_std = time_call(run_dominus, args.iterations, "DominusUltra")
    pytorch_ms, pytorch_std = time_call(run_pytorch, args.iterations, "PyTorch ref")

    out_dominus, _ = run_dominus()
    out_pytorch = run_pytorch()
    max_error = (out_dominus.float() - out_pytorch.float()).abs().max().item()

    total_tokens = args.batch_size * args.seq_len
    dominus_tps = total_tokens / (dominus_ms / 1000)
    pytorch_tps = total_tokens / (pytorch_ms / 1000)
    speedup = pytorch_ms / dominus_ms

    print(c("\nResult", BOLD + GREEN))
    print(f"DominusUltra:    {dominus_ms:8.3f} ms  {format_rate(dominus_tps):>18}")
    print(f"PyTorch ref:     {pytorch_ms:8.3f} ms  {format_rate(pytorch_tps):>18}")
    print(c(f"Speedup:         {speedup:8.2f}x", GREEN if speedup >= 1 else YELLOW))
    print(f"Max error:       {max_error:8.6f}")

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "device_name": device_name,
        "cuda_version": torch.version.cuda,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "triton_version": triton_version,
        "batch_size": args.batch_size,
        "num_heads": args.num_heads,
        "num_kv_heads": args.num_kv_heads,
        "seq_len": args.seq_len,
        "head_dim": args.head_dim,
        "dtype": args.dtype,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "dominus_ms": dominus_ms,
        "dominus_std_ms": dominus_std,
        "pytorch_ms": pytorch_ms,
        "pytorch_std_ms": pytorch_std,
        "dominus_tokens_per_sec": dominus_tps,
        "pytorch_tokens_per_sec": pytorch_tps,
        "speedup": speedup,
        "max_error": max_error,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_report(output, result)
    print(c(f"\nSaved report: {output}", CYAN))
    print(c(f"Saved data:   {output.with_suffix('.json')}", CYAN))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
