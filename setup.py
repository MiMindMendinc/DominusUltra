"""Setup script for DominusUltra package.

For modern installation, use pyproject.toml with:
    pip install -e .

This setup.py is maintained for backwards compatibility.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="dominusultra",
    version="0.1.0",
    author="Lyle Perrien II",
    author_email="contact@michiganmindmend.org",
    description="High-performance Triton kernel for fused RoPE + Grouped Query Attention (GQA) causal attention",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MiMindMendinc/DominusUltra",
    packages=find_packages(include=["dominusultra", "dominusultra.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.4.0",
        "triton>=3.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
            "numpy>=1.20",
        ],
    },
)
