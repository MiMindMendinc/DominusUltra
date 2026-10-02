# DominusUltra GPU evidence report

**Verdict:** `PASS`  
**Payload SHA-256:** `148bf0c48796376d356c59803745e3d7d04c4a684a93941e6161af8f9ba8ec2a`  
**Generated:** 2026-10-01T21:58:17.601756+00:00  
**Commit:** `a0d11750a9d5dfe858b2fa33348f8085e9bd5f2a`  
**Dirty worktree:** `False`

## Environment

- GPU: Tesla T4
- Compute capability: 7.5
- CUDA runtime: 13.0
- PyTorch: 2.11.0+cu130
- Triton: 3.6.0
- Python: 3.13.15
- Command: `/usr/bin/python3 gpu_evidence.py --suite quick --dtype auto --warmup 10 --iterations 50`

## Cases

| Case | Dtype | Status | Max error | Kernel median | Baseline median | Speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| prefill:B1:Hq8:Hkv8:T128:D64 | bfloat16 | pass | 0.000690997 | 0.2082 ms | 0.3299 ms | 1.585x |
| prefill:B1:Hq8:Hkv2:T257:D64 | bfloat16 | pass | 0.000526831 | 0.5673 ms | 0.3295 ms | 0.581x |
| decode:B2:Hq8:Hkv8:T128:D64 | bfloat16 | pass | 6.03776e-05 | 0.0922 ms | 0.2528 ms | 2.742x |
| decode:B2:Hq8:Hkv2:T257:D64 | bfloat16 | pass | 5.79488e-05 | 0.0981 ms | 0.2495 ms | 2.543x |

## Interpretation boundary

- Throughput means token positions processed by this isolated kernel call; it is not end-to-end model generation speed.
- The baseline is PyTorch SDPA with RoPE precomputed outside its timed region, a conservative comparison for a fused-RoPE kernel.
- Raw CUDA-event samples and correctness metrics are preserved in the companion JSON file.
- A PASS applies only to the recorded commit, hardware, software stack, shapes, dtypes, and tolerances.
