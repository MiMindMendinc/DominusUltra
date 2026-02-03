# dominus_ultra.py
# MIT License - Copyright (c) 2026 MiMindMendinc
"""
Dominus Ultra — fast causal attention kernel with RoPE + GQA support

Single-file Triton implementation of causal multi-query / grouped-query attention
with fused RoPE, prefill + decode paths.

Supports:
- Arbitrary head dimensions (power-of-2)
- Grouped Query Attention (GQA / MQA)
- Causal masking
- LSE output for correct incremental decoding

Requirements: torch>=2.4, triton>=3.0, CUDA sm_80+ (Ampere+)
"""

import triton
import triton.language as tl
import torch

configs = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N':  64}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M':  64, 'BLOCK_N': 128}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N':  64}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M':  64, 'BLOCK_N':  64}, num_warps=4, num_stages=5),
]

@triton.autotune(configs=configs, key=['T'])
@triton.jit
def prefill_kernel(
    Q, K, V, Cos, Sin, Out, LSE,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_ob, stride_oh, stride_ot, stride_od,
    stride_lseb, stride_lseh, stride_lset,
    B: tl.constexpr, H: tl.constexpr, KvH: tl.constexpr,
    T: tl.constexpr, D: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    kv_h = h // (H // KvH)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    q_ptrs = Q + b*stride_qb + h*stride_qh + offs_m[:,None]*stride_qt + offs_d[None,:]*stride_qd
    k_ptrs = K + b*stride_kb + kv_h*stride_kh + offs_n[None,:]*stride_kt + offs_d[:,None]*stride_kd
    v_ptrs = V + b*stride_vb + kv_h*stride_vh + offs_n[None,:]*stride_vt + offs_d[:,None]*stride_vd

    mask_q = (offs_m[:,None] < T) & (offs_d[None,:] < D)
    mask_kv = (offs_n[None,:] < T) & (offs_d[:,None] < D)

    q = tl.load(q_ptrs, mask=mask_q, other=0.0).to(tl.float32)

    half = D // 2
    q1 = q[..., :half]
    q2 = q[..., half:]
    cos = tl.load(Cos + offs_m[:,None] * half + tl.arange(0, half)[None,:],
                  mask=offs_m[:,None] < T, other=1.0)
    sin = tl.load(Sin + offs_m[:,None] * half + tl.arange(0, half)[None,:],
                  mask=offs_m[:,None] < T, other=0.0)
    q_rot = tl.where(offs_d[None,:] < half,
                     q1 * cos - q2 * sin,
                     q2 * cos + q1 * sin)

    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for start_n in range(0, T, BLOCK_N):
        offs_n_curr = start_n + offs_n

        k = tl.load(k_ptrs + start_n*stride_kt, mask=mask_kv, other=0.0).to(tl.float32)
        v = tl.load(v_ptrs + start_n*stride_vt, mask=mask_kv, other=0.0).to(tl.float32)

        k1 = k[..., :half]
        k2 = k[..., half:]
        cos_k = tl.load(Cos + offs_n_curr[None,:] * half + tl.arange(0, half)[:,None],
                        mask=(offs_n_curr[None,:] < T), other=1.0)
        sin_k = tl.load(Sin + offs_n_curr[None,:] * half + tl.arange(0, half)[:,None],
                        mask=(offs_n_curr[None,:] < T), other=0.0)
        k_rot = tl.where(tl.arange(0, D)[:,None] < half,
                         k1 * cos_k - k2 * sin_k,
                         k2 * cos_k + k1 * sin_k)

        qk = tl.dot(q_rot, tl.trans(k_rot)) * scale
        causal_mask = offs_m[:,None] >= offs_n_curr[None,:]
        qk = tl.where(causal_mask & (offs_n_curr[None,:] < T), qk, -float("inf"))

        m_ij = tl.max(qk, axis=1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)

        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float32), v)

        l_i = l_i * alpha + l_ij
        m_i = tl.maximum(m_i, m_ij)

    out = acc / l_i[:, None]
    o_ptrs = Out + b*stride_ob + h*stride_oh + offs_m[:,None]*stride_ot + offs_d[None,:]*stride_od
    tl.store(o_ptrs, out, mask=mask_q)

    lse_ptrs = LSE + b*stride_lseb + h*stride_lseh + offs_m*stride_lset
    tl.store(lse_ptrs, m_i + tl.log(l_i), mask=offs_m < T)

@triton.jit
def decode_kernel(
    Q, K_cache, V_cache, Cos, Sin, Out,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_ob, stride_oh, stride_od,
    B: tl.constexpr, H: tl.constexpr, KvH: tl.constexpr,
    past_T: tl.constexpr, D: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_N: tl.constexpr = 128
):
    pid_bh = tl.program_id(0)
    b = pid_bh // H
    h = pid_bh % H
    kv_h = h // (H // KvH)

    offs_d = tl.arange(0, D)

    q_ptr = Q + b*stride_qb + h*stride_qh + offs_d*stride_qd
    q = tl.load(q_ptr).to(tl.float32)

    half = D // 2
    q1 = q[:half]
    q2 = q[half:]
    cos_q = tl.load(Cos + past_T * half + tl.arange(0, half))
    sin_q = tl.load(Sin + past_T * half + tl.arange(0, half))
    q_rot = tl.where(offs_d < half,
                     q1 * cos_q - q2 * sin_q,
                     q2 * cos_q + q1 * sin_q)

    acc = tl.zeros((D,), dtype=tl.float32)
    m_i = -float("inf")
    l_i = 0.0

    for start_n in range(0, past_T, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        mask = offs_n < past_T

        k_ptrs = K_cache + b*stride_kb + kv_h*stride_kh + offs_n[:,None]*stride_kt + offs_d[None,:]*stride_kd
        v_ptrs = V_cache + b*stride_vb + kv_h*stride_vh + offs_n[:,None]*stride_vt + offs_d[None,:]*stride_vd

        k = tl.load(k_ptrs, mask=mask[:,None], other=0.0).to(tl.float32)
        v = tl.load(v_ptrs, mask=mask[:,None], other=0.0).to(tl.float32)

        k1 = k[:, :half]
        k2 = k[:, half:]
        cos_k = tl.load(Cos + offs_n[:,None] * half + tl.arange(0, half)[None,:],
                        mask=mask[:,None], other=1.0)
        sin_k = tl.load(Sin + offs_n[:,None] * half + tl.arange(0, half)[None,:],
                        mask=mask[:,None], other=0.0)
        k_rot = tl.where(tl.arange(0,D)[None,:] < half,
                         k1 * cos_k - k2 * sin_k,
                         k2 * cos_k + k1 * sin_k)

        qk = tl.dot(q_rot[None,:], tl.trans(k_rot)) * scale
        qk = tl.where(mask[None,:], qk, -float("inf"))

        m_ij = tl.max(qk, axis=1)[0]
        p = tl.exp(qk - m_ij)
        l_ij = tl.sum(p, axis=1)[0]

        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha + tl.dot(p[0,:].to(tl.float32), v)

        l_i = l_i * alpha + l_ij
        m_i = tl.max(m_i, m_ij)

    out = acc / l_i
    o_ptr = Out + b*stride_ob + h*stride_oh + offs_d*stride_od
    tl.store(o_ptr, out)

def dominus_ultra_prefill(q, k, v, cos, sin, num_kv_heads: int | None = None):
    B, Hq, T, D = q.shape
    _, Hk, _, _ = k.shape
    KvH = num_kv_heads if num_kv_heads is not None else Hq
    assert Hk == KvH
    assert D % 2 == 0

    out = torch.empty_like(q)
    lse = torch.empty((B, Hq, T), dtype=torch.float32, device=q.device)

    scale = 1.0 / (D ** 0.5)

    grid = lambda meta: (triton.cdiv(T, meta['BLOCK_M']), B * Hq)

    prefill_kernel[grid](
        q, k, v, cos, sin, out, lse,
        *q.stride(), *k.stride(), *v.stride(), *out.stride(), *lse.stride(),
        B=B, H=Hq, KvH=KvH, T=T, D=D, scale=scale
    )

    return out, lse

def dominus_ultra_decode(q_new, k_cache, v_cache, cos, sin, num_kv_heads: int | None = None):
    B, Hq, _, D = q_new.shape
    _, Hk, past_T, _ = k_cache.shape
    KvH = num_kv_heads if num_kv_heads is not None else Hq
    assert Hk == KvH
    assert D % 2 == 0

    out = torch.empty_like(q_new).squeeze(2)

    scale = 1.0 / (D ** 0.5)

    grid = (B * Hq,)

    decode_kernel[grid](
        q_new.squeeze(2), k_cache, v_cache, cos, sin, out,
        *q_new.squeeze(2).stride(), *k_cache.stride(), *v_cache.stride(), *out.stride(),
        B=B, H=Hq, KvH=KvH, past_T=past_T, D=D, scale=scale
    )

    return out.unsqueeze(2)

if __name__ == "__main__":
    print("Dominus Ultra loaded.")
