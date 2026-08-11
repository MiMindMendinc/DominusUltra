---
name: Benchmark result
about: Share repeatable performance results from a GPU
title: "[Benchmark]: "
labels: benchmark
assignees: ""
---

## Hardware and software

- GPU:
- Compute capability:
- CUDA:
- Driver:
- PyTorch:
- Triton:
- Python:
- OS:
- Git commit:
- Dirty worktree (`true`/`false`):

## Command

```bash
python gpu_evidence.py --suite quick --dtype auto --warmup 10 --iterations 50
```

## Evidence files

Attach the untouched files generated under `benchmark_results/`:

- JSON report:
- Markdown report:
- Payload SHA-256:

## Verdict

- [ ] PASS — every selected case passed
- [ ] FAIL — at least one case failed or raised an error

Failures are welcome. Please do not edit errors or raw timing samples out of the JSON.

## Reproduction checklist

- [ ] I ran the command from the commit named in the report.
- [ ] The report's dirty-worktree field accurately describes my checkout.
- [ ] I did not modify the generated JSON or Markdown.
- [ ] I understand that token-position throughput is not end-to-end LLM token generation speed.
- [ ] I reviewed the report and am comfortable publishing its non-secret environment metadata.

## Notes
