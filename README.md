# Dominus Ultra: Triton RoPE Causal Attention Reference Kernel

`Dominus Ultra` is an educational Triton implementation of causal attention with fused Rotary Positional Embeddings (RoPE) and Grouped Query Attention (GQA) style head mapping.

This repository is a **reference / learning project**, not a production benchmark suite. The current focus is correctness, readable Triton structure, and a small test harness that compares against PyTorch SDPA.

## What the code does today

- Fused RoPE application for Q and K inside the Triton attention kernels
- Causal attention for prefill
- A decode path for a single new token against a KV cache
- Grouped Query Attention (GQA / MQA) style mapping from query heads to KV heads
- bf16 / fp16-friendly implementation using Triton + PyTorch

## What this repository does **not** currently claim

To keep the project honest and reproducible, this repo does **not** currently claim:

- FP8 support
- Hopper TMA acceleration
- bit-perfect equivalence across all hardware
- production-readiness
- unverified throughput numbers

Those items should only be added back once they are implemented and measured with reproducible scripts.

## Requirements

- NVIDIA GPU
- CUDA-enabled PyTorch
- Python 3.10+
- `torch`
- `triton`

Example install:

```bash
git clone https://github.com/MiMindMendinc/DominusUltra.git
cd DominusUltra
pip install torch triton
```

## Files

- `dominus_ultra.py` — main Triton implementation and a small correctness check
- `README.md` — project overview

## Usage

CLI sanity check (no CUDA required):

```bash
python -m dominus_ultra --help
```

Run the included self-test (requires CUDA + a working Triton driver):

```bash
python -m dominus_ultra --self-test
```

```python
import torch
from dominus_ultra import (
    precompute_rope_cos_sin,
    dominus_ultra_prefill,
    dominus_ultra_decode,
)

device = "cuda"
dtype = torch.bfloat16

B, Hq, Hk, T, D = 1, 8, 8, 128, 64
q = torch.randn(B, Hq, T, D, device=device, dtype=dtype)
k = torch.randn(B, Hk, T, D, device=device, dtype=dtype)
v = torch.randn(B, Hk, T, D, device=device, dtype=dtype)

cos, sin = precompute_rope_cos_sin(T, D, device=device, dtype=dtype)
out, lse = dominus_ultra_prefill(q, k, v, cos, sin, num_kv_heads=Hk)
print(out.shape, lse.shape)
```

## Correctness first

The `__main__` block in `dominus_ultra.py` includes a simple reference comparison against PyTorch's `scaled_dot_product_attention` after manually applying RoPE.

Before making any performance claims, this project should:

1. pass the correctness test reliably on real hardware
2. add reproducible benchmark scripts
3. compare against PyTorch SDPA and/or FlashAttention under clearly stated settings

## Notes

This code is meant to be useful to people learning Triton attention kernels. It may still need additional fixes, tuning, and broader test coverage for different shapes and GPU generations.

## License

MIT

---

Built by Michigan MindMend Inc.
