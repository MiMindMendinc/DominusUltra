# Dominus Ultra

**Educational Triton attention-kernel project with fused RoPE and GQA-style head mapping.**

Dominus Ultra is a learning and portfolio repo for GPU-kernel experimentation. It explores causal attention, Rotary Position Embeddings (RoPE), grouped-query attention concepts, and correctness-first comparison against PyTorch reference behavior.

This repository is intentionally framed as an **experimental reference project**, not a production benchmark suite.

---

## What it does today

- Implements a Triton-based causal attention path.
- Applies RoPE to query/key tensors inside the attention workflow.
- Supports GQA/MQA-style mapping from query heads to KV heads.
- Includes a decode-style path for a single new token against a KV cache.
- Provides a small correctness check against PyTorch-style reference logic.
- Demonstrates practical GPU-kernel reading, writing, and documentation work.

---

## What it does not claim yet

To keep the project honest and recruiter-safe, this repo does **not** currently claim:

- production readiness
- Hopper TMA acceleration
- verified FP8 support
- bit-perfect equivalence across all GPU generations
- superior performance over FlashAttention
- unverified throughput numbers

Those claims should only be added if the implementation, tests, and reproducible benchmark logs support them.

---

## Why this repo matters

GPU kernels are hard to understand from blog posts alone. Dominus Ultra is a hands-on attempt to learn and document the moving parts:

```text
Q/K/V tensors
  ↓
RoPE rotation
  ↓
causal masking
  ↓
online softmax attention
  ↓
GQA/MQA head mapping
  ↓
output tensor
```

The value is in the engineering process: correctness first, readable code, then measured tuning.

---

## Requirements

- Python 3.10+
- NVIDIA GPU
- CUDA-capable PyTorch
- Triton

Example install:

```bash
git clone https://github.com/MiMindMendinc/DominusUltra.git
cd DominusUltra
pip install torch triton
```

---

## Usage example

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

---

## Correctness-first workflow

Before making performance claims, run correctness checks on real hardware and record the exact setup.

Suggested validation path:

1. Run the included reference comparison.
2. Confirm output shape and max error against a PyTorch reference.
3. Test multiple tensor shapes.
4. Test both bf16 and fp16 where supported.
5. Add a benchmark script with clear hardware details.
6. Compare against PyTorch SDPA or FlashAttention under stated settings.

---

## Recommended benchmark documentation

When benchmark results are added, include:

- GPU model
- CUDA version
- PyTorch version
- Triton version
- tensor shapes
- dtype
- warmup iterations
- measured iterations
- baseline used
- max error / tolerance

This keeps the repo credible with engineers who know GPU performance.

---

## Files

- `dominus_ultra.py` — main Triton implementation and correctness check
- `README.md` — project overview and usage notes

---

## What I built / modified

This repository demonstrates:

- Triton kernel experimentation
- attention-mechanism implementation work
- RoPE and GQA concepts
- correctness comparison against a reference
- GPU-performance documentation discipline
- willingness to revise claims based on code reality

---

## Recruiter notes

Dominus Ultra is most relevant for roles involving:

- AI systems engineering
- GPU-kernel learning
- LLM inference optimization
- PyTorch/Triton experimentation
- performance testing and benchmark hygiene

It should be read as a learning project with real technical ambition, not as a finished production kernel.

---

## Roadmap

- [ ] Add standalone test file
- [ ] Add reproducible benchmark script
- [ ] Expand shape coverage
- [ ] Add CI syntax checks where GPU is not required
- [ ] Add hardware-specific benchmark logs
- [ ] Improve comments inside the Triton kernel
- [ ] Compare against PyTorch SDPA under clear settings

---

## Built by

**Lyle Perrien II**  
Founder, **Michigan MindMend Inc.**  
Owosso, Michigan

Building privacy-first AI tools and learning the systems layer behind modern LLMs.

## License

MIT
