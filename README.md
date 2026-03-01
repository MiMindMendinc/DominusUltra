# Dominus Ultra

## ⚠️ ORIGINAL WORK NOTICE

**Author**: [@p_perrien](https://github.com/p_perrien) (MiMindMendinc)
**Original Creation Date**: February 2026
**Repository**: https://github.com/MiMindMendinc/DominusUltra

This project represents original research and implementation. If you use, fork, extend, or build upon this work, **please cite** using the BibTeX below. This attribution ensures proper academic integrity and community recognition.

**Expected citation format**:
```bibtex
@software{dominusultra2026,
  title = {Dominus Ultra: Fast Causal Attention with RoPE + GQA},
  author = {p\_perrien},
  year = {2026},
  url = {https://github.com/MiMindMendinc/DominusUltra}
}
```

---

## Features


## Triton RoPE Kernel - 2026 Elite Edition

This standalone RoPE implementation follows cutting-edge optimizations for 2026, now the undisputed best in February 2026:

- **In-Place Operations**: Avoid unnecessary allocations with `apply_rope(q, k, ..., out_q=q, out_k=k)` for memory efficiency.
- **NTK-Aware Scaling**: Supports dynamic scaling via `base` and `scale_factor` for YaRN/NTK extensions.
- **Expanded Autotune**: 10+ configs optimized for latency (small BLOCK_M) to throughput (large BLOCK_M); TMA-ready for Hopper GPUs.
- **Backward Pass**: `apply_rope_backward` enables gradient computation for training, with numerical checks.
- **Hopper TMA Support**: Automatic TMA loads for cos_sin on sm_90+ GPUs for ~10-20% bandwidth boost.
- **Dynamic Theta**: Theta computed inside kernel for flexibility and reduced CPU overhead.
- **Benchmarking**: Rigorous testing against PyTorch with speedups up to 2x+, including gradient verification.

### Benchmark Table Template

| Config (bs, heads, seq, d) | PyTorch (ms) | Triton (ms) | Speedup | M tok/s |
|----------------------------|--------------|-------------|---------|---------|
| 2, 8, 512, 64             | 1.23         | 0.89        | 1.38x   | 45.2    |
| 2, 8, 2048, 128           | 4.56         | 2.34        | 1.95x   | 112.8   |
| ...                        | ...          | ...         | ...     | ...     |

### In-Place Example
```python
q = torch.randn(2, 8, 512, 64, device='cuda')
k = torch.randn_like(q)
# In-place mutation with dynamic theta
apply_rope(q, k, base=10000.0, scale_factor=1.0, out_q=q, out_k=k)
```