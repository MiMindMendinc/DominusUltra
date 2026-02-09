"""
Comprehensive test suite for Dominus Ultra attention kernels.

Tests cover:
- Prefill kernel correctness with various configurations
- Decode kernel correctness with KV cache
- Grouped Query Attention (GQA) support
- Shape validation and error handling
- Numerical stability
"""

import pytest
import torch
import torch.nn.functional as F
from dominus_ultra import (
    dominus_ultra_prefill,
    dominus_ultra_decode,
    precompute_rope_cos_sin,
)


@pytest.fixture
def device():
    """Return CUDA device if available, else skip tests."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return "cuda"


def apply_rope_reference(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Reference implementation of RoPE for testing.
    
    Args:
        x: Input tensor of shape [..., seq_len, dim]
        cos: Cosine values of shape [seq_len, dim//2]
        sin: Sine values of shape [seq_len, dim//2]
    
    Returns:
        Tensor with RoPE applied, same shape as input
    """
    dim = x.shape[-1]
    half = dim // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class TestPrefillKernel:
    """Tests for the prefill kernel."""
    
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    @pytest.mark.parametrize("num_heads", [8, 16, 32])
    @pytest.mark.parametrize("seq_len", [64, 128, 256, 512])
    @pytest.mark.parametrize("head_dim", [64, 128])
    def test_prefill_mha_correctness(self, device, batch_size, num_heads, seq_len, head_dim):
        """Test prefill kernel correctness for Multi-Head Attention."""
        dtype = torch.bfloat16
        
        # Create input tensors
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype) * 0.1
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype) * 0.1
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype) * 0.1
        
        # Pre-compute RoPE
        cos, sin = precompute_rope_cos_sin(seq_len, head_dim, device, dtype)
        
        # Run Triton kernel
        out_triton, lse_triton = dominus_ultra_prefill(q, k, v, cos, sin, num_kv_heads=num_heads)
        
        # Reference implementation using PyTorch
        q_ref = apply_rope_reference(q.to(torch.float32), cos, sin)
        k_ref = apply_rope_reference(k.to(torch.float32), cos, sin)
        out_ref = F.scaled_dot_product_attention(
            q_ref, k_ref, v.to(torch.float32), is_causal=True
        )
        
        # Check correctness
        max_diff = (out_triton.to(torch.float32) - out_ref).abs().max().item()
        assert max_diff < 1e-2, f"Max diff {max_diff} exceeds tolerance for bf16"
    
    @pytest.mark.parametrize("num_q_heads,num_kv_heads", [(32, 8), (32, 4), (16, 4), (8, 1)])
    def test_prefill_gqa_correctness(self, device, num_q_heads, num_kv_heads):
        """Test prefill kernel with Grouped Query Attention."""
        batch_size, seq_len, head_dim = 2, 128, 64
        dtype = torch.bfloat16
        
        # Create input tensors
        q = torch.randn(batch_size, num_q_heads, seq_len, head_dim, device=device, dtype=dtype) * 0.1
        k = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device, dtype=dtype) * 0.1
        v = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device, dtype=dtype) * 0.1
        
        # Pre-compute RoPE
        cos, sin = precompute_rope_cos_sin(seq_len, head_dim, device, dtype)
        
        # Run Triton kernel
        out_triton, lse_triton = dominus_ultra_prefill(q, k, v, cos, sin, num_kv_heads=num_kv_heads)
        
        # Reference: expand KV heads to match Q heads
        group_size = num_q_heads // num_kv_heads
        k_expanded = k.repeat_interleave(group_size, dim=1)
        v_expanded = v.repeat_interleave(group_size, dim=1)
        
        q_ref = apply_rope_reference(q.to(torch.float32), cos, sin)
        k_ref = apply_rope_reference(k_expanded.to(torch.float32), cos, sin)
        out_ref = F.scaled_dot_product_attention(
            q_ref, k_ref, v_expanded.to(torch.float32), is_causal=True
        )
        
        # Check correctness
        max_diff = (out_triton.to(torch.float32) - out_ref).abs().max().item()
        assert max_diff < 1e-2, f"GQA max diff {max_diff} exceeds tolerance"
    
    def test_prefill_output_shape(self, device):
        """Test that prefill returns correct output shapes."""
        B, H, T, D = 2, 8, 128, 64
        dtype = torch.bfloat16
        
        q = torch.randn(B, H, T, D, device=device, dtype=dtype)
        k = torch.randn(B, H, T, D, device=device, dtype=dtype)
        v = torch.randn(B, H, T, D, device=device, dtype=dtype)
        cos, sin = precompute_rope_cos_sin(T, D, device, dtype)
        
        out, lse = dominus_ultra_prefill(q, k, v, cos, sin)
        
        assert out.shape == (B, H, T, D), f"Output shape mismatch: {out.shape}"
        assert lse.shape == (B, H, T), f"LSE shape mismatch: {lse.shape}"
        assert out.dtype == dtype, f"Output dtype mismatch: {out.dtype}"
        assert lse.dtype == torch.float32, f"LSE should be float32, got {lse.dtype}"
    
    def test_prefill_numerical_stability(self, device):
        """Test numerical stability with large values."""
        B, H, T, D = 1, 4, 64, 64
        dtype = torch.bfloat16
        
        # Create inputs with larger values
        q = torch.randn(B, H, T, D, device=device, dtype=dtype)
        k = torch.randn(B, H, T, D, device=device, dtype=dtype)
        v = torch.randn(B, H, T, D, device=device, dtype=dtype)
        cos, sin = precompute_rope_cos_sin(T, D, device, dtype)
        
        out, lse = dominus_ultra_prefill(q, k, v, cos, sin)
        
        # Check for NaN or Inf
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"
        assert not torch.isnan(lse).any(), "LSE contains NaN"
        assert not torch.isinf(lse).any(), "LSE contains Inf"


class TestDecodeKernel:
    """Tests for the decode kernel."""
    
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    @pytest.mark.parametrize("num_heads", [8, 16, 32])
    @pytest.mark.parametrize("past_len", [64, 128, 256])
    @pytest.mark.parametrize("head_dim", [64, 128])
    def test_decode_mha_correctness(self, device, batch_size, num_heads, past_len, head_dim):
        """Test decode kernel correctness for Multi-Head Attention."""
        dtype = torch.bfloat16
        
        # Create KV cache and new query
        k_cache = torch.randn(batch_size, num_heads, past_len, head_dim, device=device, dtype=dtype) * 0.1
        v_cache = torch.randn(batch_size, num_heads, past_len, head_dim, device=device, dtype=dtype) * 0.1
        q_new = torch.randn(batch_size, num_heads, 1, head_dim, device=device, dtype=dtype) * 0.1
        
        # Pre-compute RoPE for past_len + 1 positions
        cos, sin = precompute_rope_cos_sin(past_len + 1, head_dim, device, dtype)
        
        # Run Triton decode kernel
        out_triton = dominus_ultra_decode(q_new, k_cache, v_cache, cos, sin, num_kv_heads=num_heads)
        
        # Reference: use prefill kernel with full sequence
        # Apply RoPE to cached K
        k_ref = apply_rope_reference(k_cache.to(torch.float32), cos[:past_len], sin[:past_len])
        # Apply RoPE to new Q at position past_len
        q_ref = apply_rope_reference(q_new.to(torch.float32), cos[past_len:past_len+1], sin[past_len:past_len+1])
        
        # Compute attention for the new token
        out_ref = F.scaled_dot_product_attention(
            q_ref, k_ref, v_cache.to(torch.float32), is_causal=False
        )
        
        # Check correctness
        max_diff = (out_triton.to(torch.float32) - out_ref).abs().max().item()
        assert max_diff < 1e-2, f"Decode max diff {max_diff} exceeds tolerance"
    
    @pytest.mark.parametrize("num_q_heads,num_kv_heads", [(32, 8), (16, 4), (8, 1)])
    def test_decode_gqa_correctness(self, device, num_q_heads, num_kv_heads):
        """Test decode kernel with Grouped Query Attention."""
        batch_size, past_len, head_dim = 2, 128, 64
        dtype = torch.bfloat16
        
        # Create KV cache and new query
        k_cache = torch.randn(batch_size, num_kv_heads, past_len, head_dim, device=device, dtype=dtype) * 0.1
        v_cache = torch.randn(batch_size, num_kv_heads, past_len, head_dim, device=device, dtype=dtype) * 0.1
        q_new = torch.randn(batch_size, num_q_heads, 1, head_dim, device=device, dtype=dtype) * 0.1
        
        # Pre-compute RoPE
        cos, sin = precompute_rope_cos_sin(past_len + 1, head_dim, device, dtype)
        
        # Run Triton decode kernel
        out_triton = dominus_ultra_decode(q_new, k_cache, v_cache, cos, sin, num_kv_heads=num_kv_heads)
        
        # Reference: expand KV heads
        group_size = num_q_heads // num_kv_heads
        k_expanded = k_cache.repeat_interleave(group_size, dim=1)
        v_expanded = v_cache.repeat_interleave(group_size, dim=1)
        
        k_ref = apply_rope_reference(k_expanded.to(torch.float32), cos[:past_len], sin[:past_len])
        q_ref = apply_rope_reference(q_new.to(torch.float32), cos[past_len:past_len+1], sin[past_len:past_len+1])
        
        out_ref = F.scaled_dot_product_attention(
            q_ref, k_ref, v_expanded.to(torch.float32), is_causal=False
        )
        
        # Check correctness
        max_diff = (out_triton.to(torch.float32) - out_ref).abs().max().item()
        assert max_diff < 1e-2, f"GQA decode max diff {max_diff} exceeds tolerance"
    
    def test_decode_output_shape(self, device):
        """Test that decode returns correct output shape."""
        B, Hq, Hk, T, D = 2, 32, 8, 128, 64
        dtype = torch.bfloat16
        
        q_new = torch.randn(B, Hq, 1, D, device=device, dtype=dtype)
        k_cache = torch.randn(B, Hk, T, D, device=device, dtype=dtype)
        v_cache = torch.randn(B, Hk, T, D, device=device, dtype=dtype)
        cos, sin = precompute_rope_cos_sin(T + 1, D, device, dtype)
        
        out = dominus_ultra_decode(q_new, k_cache, v_cache, cos, sin, num_kv_heads=Hk)
        
        assert out.shape == (B, Hq, 1, D), f"Decode output shape mismatch: {out.shape}"
        assert out.dtype == dtype, f"Decode output dtype mismatch: {out.dtype}"


class TestRoPEFunction:
    """Tests for RoPE helper function."""
    
    def test_rope_precompute_shape(self, device):
        """Test RoPE precompute returns correct shapes."""
        max_seq_len, dim = 2048, 64
        cos, sin = precompute_rope_cos_sin(max_seq_len, dim, device, torch.float32)
        
        assert cos.shape == (max_seq_len, dim // 2), f"Cos shape mismatch: {cos.shape}"
        assert sin.shape == (max_seq_len, dim // 2), f"Sin shape mismatch: {sin.shape}"
    
    def test_rope_precompute_dtype(self, device):
        """Test RoPE precompute respects dtype."""
        max_seq_len, dim = 128, 64
        
        for dtype in [torch.float32, torch.bfloat16, torch.float16]:
            cos, sin = precompute_rope_cos_sin(max_seq_len, dim, device, dtype)
            assert cos.dtype == dtype, f"Cos dtype mismatch"
            assert sin.dtype == dtype, f"Sin dtype mismatch"
    
    def test_rope_values_range(self, device):
        """Test RoPE values are in valid range [-1, 1]."""
        max_seq_len, dim = 256, 128
        cos, sin = precompute_rope_cos_sin(max_seq_len, dim, device, torch.float32)
        
        assert cos.min() >= -1.0 and cos.max() <= 1.0, "Cos values out of range"
        assert sin.min() >= -1.0 and sin.max() <= 1.0, "Sin values out of range"


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_small_sequence_length(self, device):
        """Test with very small sequence lengths."""
        B, H, T, D = 1, 4, 8, 64
        dtype = torch.bfloat16
        
        q = torch.randn(B, H, T, D, device=device, dtype=dtype)
        k = torch.randn(B, H, T, D, device=device, dtype=dtype)
        v = torch.randn(B, H, T, D, device=device, dtype=dtype)
        cos, sin = precompute_rope_cos_sin(T, D, device, dtype)
        
        out, lse = dominus_ultra_prefill(q, k, v, cos, sin)
        
        assert out.shape == (B, H, T, D)
        assert not torch.isnan(out).any()
    
    def test_single_token(self, device):
        """Test decode with single token in cache."""
        B, H, D = 1, 8, 64
        dtype = torch.bfloat16
        
        q_new = torch.randn(B, H, 1, D, device=device, dtype=dtype)
        k_cache = torch.randn(B, H, 1, D, device=device, dtype=dtype)
        v_cache = torch.randn(B, H, 1, D, device=device, dtype=dtype)
        cos, sin = precompute_rope_cos_sin(2, D, device, dtype)
        
        out = dominus_ultra_decode(q_new, k_cache, v_cache, cos, sin)
        
        assert out.shape == (B, H, 1, D)
        assert not torch.isnan(out).any()
    
    def test_different_dtypes(self, device):
        """Test kernel works with different dtypes."""
        B, H, T, D = 1, 4, 64, 64
        
        for dtype in [torch.bfloat16, torch.float16]:
            q = torch.randn(B, H, T, D, device=device, dtype=dtype)
            k = torch.randn(B, H, T, D, device=device, dtype=dtype)
            v = torch.randn(B, H, T, D, device=device, dtype=dtype)
            cos, sin = precompute_rope_cos_sin(T, D, device, dtype)
            
            out, lse = dominus_ultra_prefill(q, k, v, cos, sin)
            
            assert out.dtype == dtype
            assert not torch.isnan(out).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
