"""Standalone half-split Rotary Position Embedding kernels.

This module exposes forward and explicit backward transforms for Q/K tensors.
It uses the same half-split RoPE convention as ``dominus_ultra.py`` and keeps a
small correctness-gated command-line benchmark for focused RoPE experiments.
"""

from __future__ import annotations

import argparse
import statistics
from typing import Callable, Optional, Tuple

import torch
import triton  # type: ignore[import-untyped]
import triton.language as tl  # type: ignore[import-untyped]


@triton.jit
def _rope_kernel(
    Q,
    K,
    Cos,
    Sin,
    OutQ,
    OutK,
    stride_qb,
    stride_qh,
    stride_qt,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kt,
    stride_kd,
    stride_oqb,
    stride_oqh,
    stride_oqt,
    stride_oqd,
    stride_okb,
    stride_okh,
    stride_okt,
    stride_okd,
    H: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    BACKWARD: tl.constexpr,
):
    row = tl.program_id(0)
    t = row % T
    h = (row // T) % H
    b = row // (H * T)

    offs_d = tl.arange(0, D)
    half = D // 2
    rope_d = tl.where(offs_d < half, offs_d, offs_d - half)
    partner_d = tl.where(offs_d < half, offs_d + half, offs_d - half)

    q_base = Q + b * stride_qb + h * stride_qh + t * stride_qt
    k_base = K + b * stride_kb + h * stride_kh + t * stride_kt
    q = tl.load(q_base + offs_d * stride_qd)
    k = tl.load(k_base + offs_d * stride_kd)
    q_partner = tl.load(q_base + partner_d * stride_qd)
    k_partner = tl.load(k_base + partner_d * stride_kd)

    cos = tl.load(Cos + t * half + rope_d)
    sin = tl.load(Sin + t * half + rope_d)
    if BACKWARD:
        out_q = tl.where(
            offs_d < half,
            q * cos + q_partner * sin,
            q * cos - q_partner * sin,
        )
        out_k = tl.where(
            offs_d < half,
            k * cos + k_partner * sin,
            k * cos - k_partner * sin,
        )
    else:
        out_q = tl.where(
            offs_d < half,
            q * cos - q_partner * sin,
            q * cos + q_partner * sin,
        )
        out_k = tl.where(
            offs_d < half,
            k * cos - k_partner * sin,
            k * cos + k_partner * sin,
        )

    out_q_base = OutQ + b * stride_oqb + h * stride_oqh + t * stride_oqt
    out_k_base = OutK + b * stride_okb + h * stride_okh + t * stride_okt
    tl.store(out_q_base + offs_d * stride_oqd, out_q)
    tl.store(out_k_base + offs_d * stride_okd, out_k)


def _validate_pair(first: torch.Tensor, second: torch.Tensor) -> None:
    if first.ndim != 4 or second.ndim != 4 or first.shape != second.shape:
        raise ValueError("Q and K must have the same rank-4 [B, H, T, D] shape")
    if not first.is_cuda or not second.is_cuda:
        raise ValueError("the standalone RoPE kernels require CUDA tensors")
    if first.device != second.device or first.dtype != second.dtype:
        raise ValueError("Q and K must have the same device and dtype")
    if first.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Q and K must use float16 or bfloat16")
    if not first.is_contiguous() or not second.is_contiguous():
        raise ValueError("Q and K must be contiguous")
    head_dim = first.shape[-1]
    if head_dim < 16 or head_dim > 256 or head_dim & (head_dim - 1):
        raise ValueError("head_dim must be a power of two between 16 and 256")


def _validate_output(output: torch.Tensor, reference: torch.Tensor, name: str) -> None:
    if (
        output.shape != reference.shape
        or output.device != reference.device
        or output.dtype != reference.dtype
        or not output.is_contiguous()
    ):
        raise ValueError(
            f"{name} must be contiguous and match its input shape/device/dtype"
        )


def make_half_split_pos_emb(
    seq_len: int,
    head_dim: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    base: float = 10000.0,
    scale_factor: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create half-width cosine and sine tables used by half-split RoPE."""
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if head_dim <= 0 or head_dim % 2:
        raise ValueError("head_dim must be a positive even integer")
    if base <= 0 or scale_factor <= 0:
        raise ValueError("base and scale_factor must be positive")
    inv_freq = 1.0 / (
        base
        ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    inv_freq = inv_freq * scale_factor
    angles = (
        torch.arange(seq_len, device=device, dtype=torch.float32)[:, None]
        * inv_freq[None, :]
    )
    return angles.cos().to(dtype), angles.sin().to(dtype)


def make_interleaved_pos_emb(
    seq_len: int,
    head_dim: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    base: float = 10000.0,
    scale_factor: float = 1.0,
) -> torch.Tensor:
    """Compatibility helper returning ``[cos_0, sin_0, ...]`` columns."""
    cos, sin = make_half_split_pos_emb(
        seq_len, head_dim, device, dtype, base, scale_factor
    )
    output = torch.empty((seq_len, head_dim), device=device, dtype=dtype)
    output[:, 0::2] = cos
    output[:, 1::2] = sin
    return output


def _launch(
    first: torch.Tensor,
    second: torch.Tensor,
    output_first: torch.Tensor,
    output_second: torch.Tensor,
    base: float,
    scale_factor: float,
    backward: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _validate_pair(first, second)
    _validate_output(output_first, first, "first output")
    _validate_output(output_second, second, "second output")
    B, H, T, D = first.shape
    cos, sin = make_half_split_pos_emb(
        T, D, str(first.device), first.dtype, base, scale_factor
    )
    _rope_kernel[(B * H * T,)](
        first,
        second,
        cos,
        sin,
        output_first,
        output_second,
        *first.stride(),
        *second.stride(),
        *output_first.stride(),
        *output_second.stride(),
        H=H,
        T=T,
        D=D,
        BACKWARD=backward,
    )
    return output_first, output_second


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    base: float = 10000.0,
    scale_factor: float = 1.0,
    out_q: Optional[torch.Tensor] = None,
    out_k: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply half-split RoPE to Q and K."""
    out_q = torch.empty_like(q) if out_q is None else out_q
    out_k = torch.empty_like(k) if out_k is None else out_k
    return _launch(q, k, out_q, out_k, base, scale_factor, backward=False)


def apply_rope_backward(
    grad_out_q: torch.Tensor,
    grad_out_k: torch.Tensor,
    base: float = 10000.0,
    scale_factor: float = 1.0,
    grad_q: Optional[torch.Tensor] = None,
    grad_k: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply the exact transpose rotation to Q/K output gradients."""
    grad_q = torch.empty_like(grad_out_q) if grad_q is None else grad_q
    grad_k = torch.empty_like(grad_out_k) if grad_k is None else grad_k
    return _launch(
        grad_out_q,
        grad_out_k,
        grad_q,
        grad_k,
        base,
        scale_factor,
        backward=True,
    )


def _reference(
    tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, backward: bool = False
) -> torch.Tensor:
    half = tensor.shape[-1] // 2
    low, high = tensor[..., :half], tensor[..., half:]
    if backward:
        return torch.cat((low * cos + high * sin, high * cos - low * sin), dim=-1)
    return torch.cat((low * cos - high * sin, high * cos + low * sin), dim=-1)


def _time_cuda(function: Callable[[], object], warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    starts = []
    ends = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        starts.append(start)
        ends.append(end)
    torch.cuda.synchronize()
    return statistics.median(
        start.elapsed_time(end) for start, end in zip(starts, ends)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone DominusUltra RoPE check")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is required")
        return 2
    if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        print("This GPU/runtime does not report bfloat16 support")
        return 2
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    torch.manual_seed(42)
    shape = (args.batch_size, args.heads, args.seq_len, args.head_dim)
    q = torch.randn(shape, device="cuda", dtype=dtype)
    k = torch.randn(shape, device="cuda", dtype=dtype)
    cos, sin = make_half_split_pos_emb(args.seq_len, args.head_dim, "cuda", dtype)
    q_expected = _reference(q, cos, sin)
    k_expected = _reference(k, cos, sin)
    q_actual, k_actual = apply_rope(q, k)
    tolerance = 2.0e-2 if dtype == torch.bfloat16 else 6.0e-3
    passed = torch.allclose(
        q_actual, q_expected, atol=tolerance, rtol=tolerance
    ) and torch.allclose(k_actual, k_expected, atol=tolerance, rtol=tolerance)
    max_error = max(
        (q_actual.float() - q_expected.float()).abs().max().item(),
        (k_actual.float() - k_expected.float()).abs().max().item(),
    )
    median_ms = _time_cuda(lambda: apply_rope(q, k), args.warmup, args.iterations)
    token_positions = args.batch_size * args.heads * args.seq_len
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Shape: {shape} | dtype: {args.dtype}")
    print(f"Correctness: {'PASS' if passed else 'FAIL'} | max error: {max_error:.6g}")
    print(f"Median: {median_ms:.4f} ms")
    print(f"Token positions/s: {token_positions / (median_ms / 1000.0):,.0f}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
