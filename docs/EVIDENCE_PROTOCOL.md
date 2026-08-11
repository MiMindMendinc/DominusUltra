# DominusUltra evidence protocol

DominusUltra does not treat a screenshot, a collected test count, or a speedup printed without raw context as proof. A publishable result must identify the source commit and environment, pass the numerical gates, preserve raw timing samples, and be reproducible by another GPU operator.

## Fastest independent run

Open the [Colab GPU evidence notebook](https://colab.research.google.com/github/MiMindMendinc/DominusUltra/blob/main/colab/DominusUltra_GPU_Evidence.ipynb), select a GPU runtime, and run every cell. The notebook produces a ZIP containing JSON and Markdown reports.

To run locally on an NVIDIA CUDA machine:

```bash
git clone https://github.com/MiMindMendinc/DominusUltra.git
cd DominusUltra
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"

python gpu_evidence.py --suite quick --dtype auto --warmup 10 --iterations 50
```

For a broader shape matrix and both supported low-precision dtypes where available:

```bash
python gpu_evidence.py --suite full --dtype both --warmup 20 --iterations 100
```

Submit the untouched JSON and Markdown files through the repository's [Benchmark result issue](https://github.com/MiMindMendinc/DominusUltra/issues/new?template=benchmark_result.md). Failed compile, correctness, and runtime reports are valuable evidence too.

## What the runner records

- exact Git commit, branch, and dirty-worktree status;
- SHA-256 hashes of the kernel, runner, and dependency file;
- GPU name, compute capability, memory, CUDA/driver data, and relevant framework versions;
- fixed seed, shapes, dtype, warmup count, iteration count, and tolerances;
- maximum/mean absolute error and RMSE against a readable PyTorch reference;
- prefill LSE error against a direct log-sum-exp reference;
- every CUDA-event latency sample, not only an average;
- median, mean, p95, standard deviation, throughput, and baseline ratio;
- a SHA-256 digest over the canonical JSON payload.

The process exits nonzero if any selected case fails to compile, raises at runtime, or exceeds its correctness tolerance. A performance number from a failed report must not be promoted.

## Comparison boundary

The comparison baseline is PyTorch scaled-dot-product attention with RoPE precomputed outside the timed region. DominusUltra performs RoPE inside the timed fused kernel, so the baseline is intentionally conservative with respect to RoPE overhead.

Reported throughput is isolated **token-position throughput** for the selected attention call. It is not end-to-end language-model generation speed. Results apply only to the recorded commit, device, software stack, shapes, dtypes, and tolerances.

## Evidence levels

| Level | Meaning | Public claim allowed |
| --- | --- | --- |
| 0 | Source or static review only | Implementation exists; GPU status unverified |
| 1 | Screenshot without complete metadata/raw data | Preliminary operator capture only |
| 2 | Correctness-gated raw report from one clean commit | Verified on the recorded configuration |
| 3 | Independent passing reports from at least two GPU architectures | Reproduced across the named architectures |

The project remains at Level 1 for the legacy standalone RoPE capture until the fused prefill/decode runner produces an untouched passing report. Level 2 is the immediate release gate.

## Verify the payload digest

```bash
python - your-report.json <<'PY'
import hashlib
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
payload = json.dumps(
    report["payload"], sort_keys=True, separators=(",", ":"), ensure_ascii=True
).encode("utf-8")
actual = hashlib.sha256(payload).hexdigest()
print("PASS" if actual == report["payload_sha256"] else "FAIL", actual)
PY
```

The digest detects accidental or casual editing after generation. It is not a digital signature and does not prove who ran the benchmark.
