# DominusUltra GPU evidence report

**Verdict:** `PASS`  
**Payload SHA-256:** `023bd92ac67af5bb10b0d5932fe649c60ee9620a9475d9f1be7c15899c8fd32d`  
**Generated:** 2026-10-01T23:14:00.033434+00:00  
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
| prefill:B1:Hq8:Hkv8:T128:D64 | bfloat16 | pass | 0.000690997 | 0.1618 ms | 0.3604 ms | 2.228x |
| prefill:B1:Hq8:Hkv2:T257:D64 | bfloat16 | pass | 0.000526831 | 3.4888 ms | 0.3515 ms | 0.101x |
| decode:B2:Hq8:Hkv8:T128:D64 | bfloat16 | pass | 6.03776e-05 | 0.1469 ms | 0.4363 ms | 2.970x |
| decode:B2:Hq8:Hkv2:T257:D64 | bfloat16 | pass | 5.79488e-05 | 0.2067 ms | 0.4150 ms | 2.008x |

## Interpretation boundary

- Throughput means token positions processed by this isolated kernel call; it is not end-to-end model generation speed.
- The baseline is PyTorch SDPA with RoPE precomputed outside its timed region, a conservative comparison for a fused-RoPE kernel.
- Raw CUDA-event samples and correctness metrics are preserved in the companion JSON file.
- A PASS applies only to the recorded commit, hardware, software stack, shapes, dtypes, and tolerances.
