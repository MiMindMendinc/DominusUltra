# DominusUltra

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![Triton](https://img.shields.io/badge/Triton-3.x-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)

**High-performance Triton kernel for fused RoPE + Grouped Query Attention (GQA) causal attention.**

A clean, readable, reference/educational implementation designed for easy integration into any PyTorch LLM training or inference stack.

### Features
- Fused Rotary Position Embeddings (RoPE)
- Grouped Query Attention (GQA) support
- Causal attention masking
- Triton-optimized kernels
- Clear, well-documented code with correctness examples

### Installation
```bash
git clone https://github.com/MiMindMendinc/DominusUltra.git
cd DominusUltra
pip install -e .
```

### Quick Start
```python
import torch
from dominusultra import precompute_rope_cos_sin, dominus_ultra_prefill

# See examples/ for full working scripts
```

### Commercial Use & Leasing
This project is released under the MIT License — you are free to use, modify, and sell it commercially with no restrictions.

For enterprise support, custom licensing, paid maintenance, leasing arrangements, or integration help, contact: **Michigan MindMend Inc. (Lyle Perrien II)** via LinkedIn or michiganmindmend.org.

### License
MIT — see [LICENSE](LICENSE)

---

**Built and maintained by Lyle Perrien II – Founder, Michigan MindMend Inc.**
