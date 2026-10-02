# DominusUltra GPU evidence report

**Verdict:** `PASS`  
**Payload SHA-256:** `2596efb9fd77442a53cb4fa32422371914b9a6008ff0e51e230bcdbef4acfdf7`  
**Generated:** 2026-10-01T22:45:37.321539+00:00  
**Commit:** `a0d11750a9d5dfe858b2fa33348f8085e9bd5f2a`  
**Dirty worktree:** `True`

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
| prefill:B1:Hq8:Hkv8:T128:D64 | bfloat16 | pass | 0.000690997 | 0.2794 ms | 0.5000 ms | 1.790x |
| prefill:B1:Hq8:Hkv2:T257:D64 | bfloat16 | pass | 0.000526831 | 0.3870 ms | 0.5756 ms | 1.487x |
| decode:B2:Hq8:Hkv8:T128:D64 | bfloat16 | pass | 6.03776e-05 | 0.1595 ms | 0.3730 ms | 2.339x |
| decode:B2:Hq8:Hkv2:T257:D64 | bfloat16 | pass | 5.79488e-05 | 0.1692 ms | 0.3727 ms | 2.203x |

## Interpretation boundary

- Throughput means token positions processed by this isolated kernel call; it is not end-to-end model generation speed.
- The baseline is PyTorch SDPA with RoPE precomputed outside its timed region, a conservative comparison for a fused-RoPE kernel.
- Raw CUDA-event samples and correctness metrics are preserved in the companion JSON file.
- A PASS applies only to the recorded commit, hardware, software stack, shapes, dtypes, and tolerances.
