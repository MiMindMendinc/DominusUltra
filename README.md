# DominusUltra

DominusUltra is a Triton CUDA research kernel for fused-RoPE causal attention and GQA/MQA, with **142 parametrized CUDA correctness cases** collected by `pytest`.

[![CI](https://github.com/MiMindMendinc/DominusUltra/actions/workflows/ci.yml/badge.svg)](https://github.com/MiMindMendinc/DominusUltra/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](setup.py)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-ee4c2c.svg)](requirements.txt)
[![Triton](https://img.shields.io/badge/Triton-3.0%2B-111111.svg)](requirements.txt)

## Why it matters

Production attention libraries are intentionally opaque at the kernel boundary. This repository keeps prefill, decode, rotary embeddings, grouped-query head mapping, and the PyTorch reference close enough to read together. It is intended for correctness work and controlled CUDA experiments, not as a drop-in replacement for a production attention library.

## Benchmarks

`benchmark.py` compares the Triton path with this repository's unfused PyTorch reference: RoPE is applied in PyTorch, then `torch.nn.functional.scaled_dot_product_attention` runs on the same tensor shapes. The default prefill sweep uses batch size 2, 32 query/KV heads, head dimension 64, sequence lengths 128–2048, and `bfloat16`; it also measures GQA at sequence length 1024. The decode sweep uses batch size 8, 32 heads, head dimension 64, and cache lengths 128–2048.

```bash
python benchmark.py --mode all --dtype bfloat16
```

The harness prints CUDA, PyTorch, dtype, latency, and speedup context. No reviewer-ready performance result is currently committed: the earlier `7x` / `~1.8 TB/s` row was removed because no raw result accompanied it and this harness does not calculate effective bandwidth. Commit the raw output, GPU model, driver/runtime, PyTorch version, Triton version, dtype, and exact command before quoting a number.

For a report-producing run:

```bash
python demo_speedtest.py --seq-len 2048 --dtype bfloat16 --iterations 40
```

This writes Markdown and JSON under `benchmark_results/` after checking numerical error. CUDA is required; the command exits with status 2 when no NVIDIA CUDA device is available.

## Install and quickstart

```bash
git clone https://github.com/MiMindMendinc/DominusUltra.git
cd DominusUltra
python -m venv .venv
```

Activate the environment:

```bash
# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.\.venv\Scripts\activate
```

Install the package and development dependencies:

```bash
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Run a minimal prefill call on a CUDA GPU:

```python
import torch
from dominus_ultra import dominus_ultra_prefill, precompute_rope_cos_sin

q = torch.randn(1, 8, 512, 64, device="cuda", dtype=torch.bfloat16)
k = torch.randn(1, 8, 512, 64, device="cuda", dtype=torch.bfloat16)
v = torch.randn(1, 8, 512, 64, device="cuda", dtype=torch.bfloat16)

cos, sin = precompute_rope_cos_sin(512, 64, device="cuda", dtype=torch.bfloat16)
out, lse = dominus_ultra_prefill(q, k, v, cos, sin, num_kv_heads=8)

print(out.shape)  # torch.Size([1, 8, 512, 64])
print(lse.shape)  # torch.Size([1, 8, 512])
```

Requirements: Python 3.8+, PyTorch 2.4+, Triton 3.0+, and an NVIDIA CUDA GPU. Ampere or newer is recommended for the included kernel configurations.

## Test suite

```bash
pytest -q
```

`pytest` collects **142 cases** covering prefill, decode, GQA/MQA head layouts, output shape and dtype, RoPE ranges, numerical stability, and edge shapes. Every case requires CUDA. A CPU-only run therefore reports `142 skipped`; that is an environment limitation, not passing GPU evidence. A reviewer-ready release still needs a recorded CUDA run with the device and dependency versions.

Static verification used during review:

```bash
ruff check .
mypy dominus_ultra.py rope.py benchmark.py demo_speedtest.py
python -m compileall -q dominus_ultra.py rope.py benchmark.py demo_speedtest.py
```

## Architecture

- `dominus_ultra.py` contains the fused-RoPE prefill kernel, decode kernel, and Python launch wrappers.
- `test_dominus.py` defines the PyTorch-reference correctness contract.
- `benchmark.py` runs synchronized latency comparisons for MHA and GQA shapes.
- `demo_speedtest.py` records hardware metadata, latency, speedup, and maximum numerical error.
- `rope.py` contains the standalone forward/backward RoPE experiment.
- `examples/webgpu-rope-demo.html` provides a browser-side RoPE visualization and CPU reference timing.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the tiled prefill/decode data flow and [docs/RECORDING_GUIDE.md](docs/RECORDING_GUIDE.md) for benchmark capture guidance.

## Correctness and limitations

- The CUDA tests compare outputs with a readable PyTorch reference using dtype-aware tolerances.
- The benchmark baseline is this repository's PyTorch reference, not FlashAttention or another fused library.
- The kernels are research code and have not received an independent security or production-readiness audit.
- Performance depends on GPU architecture, driver/runtime, dtype, shape, and installed PyTorch/Triton versions.

A fused-RoPE experiment related to this work was submitted to `xai-org/grok-1` as [PR #434](https://github.com/xai-org/grok-1/pull/434). That pull request is separate from this repository's correctness and benchmark evidence.

## License

[MIT](LICENSE). See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for project policies.
