# DominusUltra GPU evidence report

**Verdict:** `PASS`  
**Payload SHA-256:** `fb22b81f5135c1a73db1ac260acf71b1a539a0bcb5f72ccbd8fa6570f3e071b8`  
**Generated:** 2026-10-02T02:33:10.777876+00:00  
**Commit:** `a0d11750a9d5dfe858b2fa33348f8085e9bd5f2a`  
**Dirty worktree:** `True`

## Environment

- GPU: Tesla T4
- Compute capability: 7.5
- CUDA runtime: 13.0
- PyTorch: 2.11.0+cu130
- Triton: 3.6.0
- Python: 3.13.15
- Command: `/usr/bin/python3 gpu_evidence.py --suite full --dtype auto --warmup 10 --iterations 50`

## Cases

| Case | Dtype | Status | Max error | Kernel median | Baseline median | Speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| prefill:B1:Hq8:Hkv8:T128:D64 | bfloat16 | pass | 0.000690997 | 0.2884 ms | 0.3245 ms | 1.125x |
| prefill:B1:Hq8:Hkv2:T257:D64 | bfloat16 | pass | 0.000526831 | 0.5673 ms | 0.3172 ms | 0.559x |
| decode:B2:Hq8:Hkv8:T128:D64 | bfloat16 | pass | 6.03776e-05 | 0.0931 ms | 0.2470 ms | 2.653x |
| decode:B2:Hq8:Hkv2:T257:D64 | bfloat16 | pass | 5.79488e-05 | 0.0967 ms | 0.2325 ms | 2.404x |
| prefill:B1:Hq16:Hkv16:T512:D64 | bfloat16 | pass | 0.000698313 | 6.3700 ms | 1.0432 ms | 0.164x |
| prefill:B1:Hq16:Hkv4:T1024:D64 | bfloat16 | pass | 0.000647649 | 26.0750 ms | 4.2130 ms | 0.162x |
| prefill:B1:Hq8:Hkv2:T512:D128 | bfloat16 | pass | 0.000607789 | 11.7882 ms | 0.6738 ms | 0.057x |
| decode:B4:Hq16:Hkv16:T512:D64 | bfloat16 | pass | 3.09777e-05 | 0.1022 ms | 0.2943 ms | 2.879x |
| decode:B4:Hq16:Hkv4:T1024:D64 | bfloat16 | pass | 2.92445e-05 | 0.1085 ms | 0.5575 ms | 5.136x |
| decode:B2:Hq8:Hkv2:T512:D128 | bfloat16 | pass | 3.06945e-05 | 0.1890 ms | 0.2561 ms | 1.355x |

## Interpretation boundary

- Throughput means token positions processed by this isolated kernel call; it is not end-to-end model generation speed.
- The baseline is PyTorch SDPA with RoPE precomputed outside its timed region, a conservative comparison for a fused-RoPE kernel.
- Raw CUDA-event samples and correctness metrics are preserved in the companion JSON file.
- A PASS applies only to the recorded commit, hardware, software stack, shapes, dtypes, and tolerances.
