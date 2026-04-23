"""
Backwards compatibility shim for dominus_ultra.py

This file is maintained for backwards compatibility.
New code should import from the dominusultra package:
    from dominusultra import precompute_rope_cos_sin, dominus_ultra_prefill, dominus_ultra_decode
"""

# Import everything from the package
from dominusultra import *  # noqa: F401, F403

# Preserve backwards compatibility
__all__ = [
    "precompute_rope_cos_sin",
    "dominus_ultra_prefill",
    "dominus_ultra_decode",
]
