# DominusUltra

Fast Triton causal-attention kernels with fused RoPE, GQA/MQA support, and decode-time KV-cache paths for LLM inference experiments.

[![CI](https://github.com/MiMindMendinc/DominusUltra/actions/workflows/ci.yml/badge.svg)](https://github.com/MiMindMendinc/DominusUltra/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](setup.py)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-ee4c2c.svg)](requirements.txt)
[![Triton](https://img.shields.io/badge/Triton-3.0%2B-111111.svg)](requirements.txt)

DominusUltra is a single-file, readable Triton implementation of causal attention for modern decoder-only LLM workloads. It focuses on the pieces that matter in real inference systems: rotary position embeddings, grouped-query attention, numerically stable causal masking, and separate prefill/decode paths.

> Status: alpha research kernel. The code is meant to be studied, benchmarked, and extended on CUDA GPUs, especially NVIDIA Ampere or newer.

## Why It Matters

Most LLM projects call optimized attention as a black box. DominusUltra opens that box. It shows how attention kernels are built from the inside: tiling, online softmax, RoPE fusion, GQA head mapping, and KV-cache decode behavior.

This repository is useful if you want to:

- Learn how Triton attention kernels are structured.
- Compare a fused RoPE attention path against PyTorch reference behavior.
- Experiment with GQA/MQA layouts used by modern LLMs.
- Build a portfolio artifact for LLM inference, GPU programming, or systems AI roles.

## Features

| Area | What is included |
| --- | --- |
| Prefill | Fused causal attention kernel with RoPE applied inside the Triton path |
| Decode | Single-token decode path with KV-cache inputs |
| Attention layouts | MHA, GQA, and MQA-style KV head sharing |
| Correctness | PyTorch reference comparisons in `test_dominus.py` |
| Benchmarking | Reproducible benchmark harness in `benchmark.py` |
| Packaging | Editable install via `setup.py` and `requirements.txt` |

## Installation

```bash
git clone https://github.com/MiMindMendinc/DominusUltra.git
cd DominusUltra
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -e ".[dev]"
```

Linux/macOS activation:

```bash
source .venv/bin/activate
```

## Quick Start

```python
import torch
from dominus_ultra import dominus_ultra_prefill, precompute_rope_cos_sin

q = torch.randn(1, 8, 512, 64, device="cuda", dtype=torch.bfloat16)
k = torch.randn(1, 8, 512, 64, device="cuda", dtype=torch.bfloat16)
v = torch.randn(1, 8, 512, 64, device="cuda", dtype=torch.bfloat16)

cos, sin = precompute_rope_cos_sin(512, 64, device="cuda", dtype=torch.bfloat16)
out, lse = dominus_ultra_prefill(q, k, v, cos, sin, num_kv_heads=8)

print(out.shape)
print(lse.shape)
```

## Validate Correctness

The pytest suite compares Triton outputs with PyTorch reference implementations across prefill, decode, GQA, shape validation, and numerical stability cases.

```bash
pytest -q
```

CUDA is required for the kernel tests. On CPU-only machines the tests are skipped instead of producing false failures.

## Run Benchmarks

```bash
python benchmark.py --mode all --dtype bfloat16
```

Useful targeted runs:

```bash
python benchmark.py --mode prefill --dtype bfloat16
python benchmark.py --mode decode --dtype float16
```

The benchmark prints Triton latency, PyTorch reference latency, and speedup for MHA and GQA configurations. When sharing results, include GPU model, CUDA version, PyTorch version, Triton version, dtype, and command used.

## Project Layout

```text
DominusUltra/
  dominus_ultra.py              # Main Triton kernels and Python wrappers
  rope.py                       # RoPE-focused helper/reference work
  test_dominus.py               # CUDA correctness tests
  benchmark.py                  # Performance harness
  examples/webgpu-rope-demo.html
  setup.py
  requirements.txt
  CONTRIBUTING.md
  SECURITY.md
  CODE_OF_CONDUCT.md
```

## What To Look At First

- `dominus_ultra.py`: the core kernel implementation.
- `test_dominus.py`: the correctness contract and supported shapes.
- `benchmark.py`: the performance story and repeatable measurement path.
- `examples/webgpu-rope-demo.html`: browser-side RoPE visualization/demo material.

## Roadmap

- [x] Fused RoPE prefill kernel
- [x] GQA/MQA KV head sharing
- [x] Decode path with KV cache
- [x] Benchmark harness
- [ ] Publish benchmark table with GPU-specific results
- [ ] Add FlashAttention-style tiling experiments
- [ ] Add FP8/INT8 experimentation branch
- [ ] Expand WebGPU/browser demo work

## Contributing

Contributions are welcome, especially benchmark results, correctness tests, documentation improvements, and kernel experiments. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

## Author

Built by Lyle Perrien / MiMindMend Inc. as part of a broader private, high-performance AI inference stack.

MIT License. See [LICENSE](LICENSE).
