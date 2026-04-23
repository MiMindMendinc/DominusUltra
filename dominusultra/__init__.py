"""
DominusUltra - High-performance Triton kernel for fused RoPE + GQA causal attention

A clean, readable, reference/educational implementation designed for easy integration
into any PyTorch LLM training or inference stack.

MIT License - Copyright (c) 2026 Michigan MindMend Inc.
"""

__version__ = "0.1.0"
__author__ = "Lyle Perrien II"
__email__ = "contact@michiganmindmend.org"

from .core import (
    precompute_rope_cos_sin,
    dominus_ultra_prefill,
    dominus_ultra_decode,
)

__all__ = [
    "precompute_rope_cos_sin",
    "dominus_ultra_prefill",
    "dominus_ultra_decode",
    "__version__",
]
