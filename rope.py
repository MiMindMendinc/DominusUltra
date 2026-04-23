"""
Backwards compatibility shim for rope.py

This file is maintained for backwards compatibility.
New code should import from the dominusultra package:
    from dominusultra.rope import ...
"""

# Import everything from the package
from dominusultra.rope import *  # noqa: F401, F403
