# DominusUltra

DominusUltra is a Triton CUDA research kernel for fused-RoPE causal attention and GQA/MQA. The repository includes a **142-case CUDA correctness matrix**, seven CPU RoPE contract cases, and a report generator that refuses to label a GPU run successful unless every selected case passes its numerical gate.

[![CI](https://github.com/MiMindMendinc/DominusUltra/actions/workflows/ci.yml/badge.svg)](https://github.com/MiMindMendinc/DominusUltra/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](setup.py)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-ee4c2c.svg)](requirements.txt)
[![Triton](https://img.shields.io/badge/Triton-3.0%2B-111111.svg)](requirements.txt)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MiMindMendinc/DominusUltra/blob/main/colab/DominusUltra_GPU_Evidence.ipynb)

> **CI scope:** Green CI is CPU/contract. GPU proof is the gated Colab artifacts under [`docs/evidence/`](docs/evidence/) — see [Latest gated run](#latest-gated-run-tesla-t4).

## Latest gated run (Tesla T4)

- **GPU:** Tesla T4 sm_75 · Triton 3.6.0 · PyTorch 2.11.0+cu130
- **Pinned commit:** `a0d11750a9d5dfe858b2fa33348f8085e9bd5f2a`
- **Clean quick (release):** [`…20261002T022606Z.json`](docs/evidence/raw/dominus-ultra-evidence-20261002T022606Z.json) · `dirty=false` · payload SHA-256 `35981c65d2ce2c6b18a790a8f4a8ce8143683b4aacc0481a0e41154a0a17ad02` · verdict **PASS** · decode **2.503×** on `decode:B2:Hq8:Hkv8:T128:D64`, prefill **0.451×** loss on `prefill:B1:Hq8:Hkv2:T257:D64`. Quote both.
- **Full suite:** [`…20261002T023310Z.json`](docs/evidence/raw/dominus-ultra-evidence-20261002T023310Z.json) · 10/10 **PASS** · worktree `dirty=true` (prior evidence files only; `dominus_ultra.py` SHA matches clean) · best **5.136×** on `decode:B4:Hq16:Hkv4:T1024:D64` and worst **0.057×** on `prefill:B1:Hq8:Hkv2:T512:D128`. Short-cache decode wins are launch-overhead, not a bandwidth claim. Prefill losses are real.
- **Summary:** [`docs/evidence/T4_2026-10-02.md`](docs/evidence/T4_2026-10-02.md) · prior day [`T4_2026-10-01.md`](docs/evidence/T4_2026-10-01.md) · [#17](https://github.com/MiMindMendinc/DominusUltra/issues/17)
- **Baseline:** unfused PyTorch RoPE + `scaled_dot_product_attention` — not FlashAttention

## Why it matters

Production attention libraries are intentionally opaque at the kernel boundary. This repository keeps prefill, decode, rotary embeddings, grouped-query head mapping, and the PyTorch reference close enough to read together. It is intended for correctness work and controlled CUDA experiments, not as a drop-in replacement for a production attention library.

## Benchmarks

`benchmark.py` compares the Triton path with this repository's unfused PyTorch reference: RoPE is applied in PyTorch, then `torch.nn.functional.scaled_dot_product_attention` runs on the same tensor shapes. The default prefill sweep uses batch size 2, 32 query/KV heads, head dimension 64, sequence lengths 128–2048, and `bfloat16`; it also measures GQA at sequence length 1024. The decode sweep uses batch size 8, 32 heads, head dimension 64, and cache lengths 128–2048.

```bash
python benchmark.py --mode all --dtype bfloat16
```

The reviewer-ready result is the gated T4 report above, not a single speedup. Do not quote 2.503× or 5.136× without the matching prefill loss (0.451× quick, 0.057× full). The earlier `7x` / `~1.8 TB/s` row stays removed: no raw result accompanied it, and this repository does not calculate effective bandwidth.

A preliminary [operator-supplied Colab CUDA verification capture](docs/evidence/COLAB_T4_ROPE_8192.md) is preserved with its exact visible numbers and limitations. It covers a standalone RoPE experiment—not the fused prefill/decode kernel—and is not promoted as reviewer-complete evidence because the screenshot omits required environment metadata and raw output.

For the release-gating quick matrix:

```bash
python gpu_evidence.py --suite quick --dtype auto --warmup 10 --iterations 50
```

For a broader matrix across both supported low-precision dtypes where the GPU supports them:

```bash
python gpu_evidence.py --suite full --dtype both --warmup 20 --iterations 100
```

The runner writes Markdown and raw JSON under `benchmark_results/`. It records the exact commit, source hashes, GPU/software stack, correctness error, LSE error, every CUDA-event sample, summary statistics, and a SHA-256 payload digest. It exits nonzero on compile errors, runtime failures, or tolerance failures. See the [evidence protocol](docs/EVIDENCE_PROTOCOL.md), or use the one-click Colab badge above and submit either a passing or failing report through the [benchmark issue template](https://github.com/MiMindMendinc/DominusUltra/issues/new?template=benchmark_result.md).

`demo_speedtest.py` remains available for a recordable single-shape terminal demo, but its output is not the release gate.

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

`pytest` collects **142 CUDA cases** covering prefill, decode, GQA/MQA head layouts, output shape and dtype, RoPE ranges, numerical stability, and edge shapes, plus seven CPU cases for the RoPE table contract, standalone/fused consistency, and invalid inputs. A CPU-only run can validate those seven contracts but skips the fused kernels; that is not passing GPU evidence. The gated T4 PASS is the current GPU evidence; CI does not run those kernels.

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
- `gpu_evidence.py` runs the correctness-gated, metadata-complete public evidence matrix.
- `demo_speedtest.py` records hardware metadata, latency, speedup, and maximum numerical error.
- `rope.py` contains the standalone forward/backward RoPE experiment.
- `examples/webgpu-rope-demo.html` provides a browser-side RoPE visualization and CPU reference timing.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the tiled prefill/decode data flow and [docs/RECORDING_GUIDE.md](docs/RECORDING_GUIDE.md) for benchmark capture guidance.

## Correctness and limitations

- The CUDA tests compare outputs with a readable PyTorch reference using dtype-aware tolerances.
- Source-level repairs for the RoPE table shape, half-rotation indexing, decode reduction, KV tail masking, and online-softmax rescaling have a T4 gated PASS on `a0d11750` ([#17](https://github.com/MiMindMendinc/DominusUltra/issues/17)). Issue [#4](https://github.com/MiMindMendinc/DominusUltra/issues/4) can close on that report; a second GPU is optional, not required to quote the table.
- The benchmark baseline is this repository's PyTorch reference, not FlashAttention or another fused library.
- The kernels are research code and have not received an independent security or production-readiness audit.
- Performance depends on GPU architecture, driver/runtime, dtype, shape, and installed PyTorch/Triton versions.

A fused-RoPE experiment related to this work was submitted to `xai-org/grok-1` as [PR #434](https://github.com/xai-org/grok-1/pull/434). That pull request is separate from this repository's correctness and benchmark evidence.

## License

[MIT](LICENSE). See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for project policies.
