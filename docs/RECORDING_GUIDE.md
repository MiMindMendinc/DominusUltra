# Recording Guide: Live Speed Test

Use `demo_speedtest.py` when you want a clean, recordable benchmark that looks like a live speed test and produces a shareable result afterward.

## What To Say

Short version:

> "This is a live attention-kernel speed test. It compares my fused Triton RoPE + GQA kernel against a PyTorch reference path on the same tensors, same dtype, same GPU. It warms up first, times repeated CUDA-synchronized runs, checks numerical error, then saves the result."

## Before Recording

1. Open a fresh terminal.
2. Make the terminal font large enough to read on video.
3. Close unrelated windows and background GPU workloads.
4. Activate the environment for the repo.
5. Run one practice pass before recording.

Windows:

```bash
cd C:\Users\petep\OneDrive\mindmend_v9.py\DominusUltra
.\.venv\Scripts\activate
python demo_speedtest.py --seq-len 1024 --iterations 40
```

Linux:

```bash
cd DominusUltra
source .venv/bin/activate
python demo_speedtest.py --seq-len 1024 --iterations 40
```

## Strong Demo Commands

Fast recording pass:

```bash
python demo_speedtest.py --seq-len 512 --iterations 25
```

Main client demo:

```bash
python demo_speedtest.py --seq-len 1024 --iterations 40
```

Heavier run if the GPU is strong:

```bash
python demo_speedtest.py --seq-len 2048 --iterations 30
```

## What The Viewer Sees

- Hardware and software printed up front.
- A countdown so the recording has a clean start.
- Warmup progress.
- Live DominusUltra timing.
- Live PyTorch reference timing.
- Final latency, throughput, speedup, and max error.
- A saved Markdown report in `benchmark_results/`.

## Credibility Notes

Do not promise a fixed speedup before running. Say:

> "The number depends on the GPU and shape. The point is that the method is reproducible: same tensors, same GPU, synchronized timing, and a correctness check."

That sounds professional and protects you from overclaiming.
