"""Generate reproducible correctness and performance evidence for DominusUltra.

The runner records the source commit, environment, raw latency samples, numerical
error, and a SHA-256 digest of the report payload. A report is successful only
when every selected kernel case compiles, runs, and passes its correctness gate.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import shlex
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F
import triton  # type: ignore[import-untyped]

from dominus_ultra import (
    dominus_ultra_decode,
    dominus_ultra_prefill,
    precompute_rope_cos_sin,
)


SCHEMA_VERSION = "dominus-ultra-evidence-v1"
REPOSITORY = "https://github.com/MiMindMendinc/DominusUltra"


@dataclass(frozen=True)
class EvidenceCase:
    mode: str
    batch_size: int
    num_heads: int
    num_kv_heads: int
    context_length: int
    head_dim: int

    @property
    def label(self) -> str:
        return (
            f"{self.mode}:B{self.batch_size}:Hq{self.num_heads}:"
            f"Hkv{self.num_kv_heads}:T{self.context_length}:D{self.head_dim}"
        )


QUICK_CASES: Tuple[EvidenceCase, ...] = (
    EvidenceCase("prefill", 1, 8, 8, 128, 64),
    EvidenceCase("prefill", 1, 8, 2, 257, 64),
    EvidenceCase("decode", 2, 8, 8, 128, 64),
    EvidenceCase("decode", 2, 8, 2, 257, 64),
)

FULL_CASES: Tuple[EvidenceCase, ...] = QUICK_CASES + (
    EvidenceCase("prefill", 1, 16, 16, 512, 64),
    EvidenceCase("prefill", 1, 16, 4, 1024, 64),
    EvidenceCase("prefill", 1, 8, 2, 512, 128),
    EvidenceCase("decode", 4, 16, 16, 512, 64),
    EvidenceCase("decode", 4, 16, 4, 1024, 64),
    EvidenceCase("decode", 2, 8, 2, 512, 128),
)


def apply_rope_reference(
    tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply the same half-split RoPE convention used by the Triton kernels."""
    half = tensor.shape[-1] // 2
    low = tensor[..., :half]
    high = tensor[..., half:]
    return torch.cat((low * cos - high * sin, high * cos + low * sin), dim=-1)


def _command_output(command: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return "unavailable"
    output = completed.stdout.strip() or completed.stderr.strip()
    return output if output else "unavailable"


def _git_metadata() -> Dict[str, Any]:
    commit = _command_output(("git", "rev-parse", "HEAD"))
    branch = _command_output(("git", "branch", "--show-current"))
    status = _command_output(("git", "status", "--porcelain"))
    return {
        "commit": commit,
        "branch": branch,
        "dirty": status not in ("", "unavailable"),
    }


def _source_hashes(paths: Iterable[Path]) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    for path in paths:
        if path.is_file():
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _environment_metadata(command: str) -> Dict[str, Any]:
    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    tracked_environment = {
        key: os.environ.get(key)
        for key in (
            "CUDA_VISIBLE_DEVICES",
            "CUBLAS_WORKSPACE_CONFIG",
            "NVIDIA_TF32_OVERRIDE",
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
        )
        if os.environ.get(key) is not None
    }
    sm_count = getattr(properties, "multi_processor_count", None)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repository": REPOSITORY,
        "command": command,
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_git_version": getattr(torch.version, "git_version", "unknown"),
        "triton": getattr(triton, "__version__", "unknown"),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device_index": device_index,
        "device_name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "total_memory_bytes": properties.total_memory,
        "multiprocessor_count": sm_count,
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "nvidia_smi": _command_output(
            (
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            )
        ),
        "tracked_environment": tracked_environment,
        "git": _git_metadata(),
        "source_sha256": _source_hashes(
            (
                Path("dominus_ultra.py"),
                Path("gpu_evidence.py"),
                Path("requirements.txt"),
            )
        ),
    }


def _dtype_names(selection: str) -> Tuple[str, ...]:
    if selection == "auto":
        return ("bfloat16",) if torch.cuda.is_bf16_supported() else ("float16",)
    if selection == "both":
        names = ["float16"]
        if torch.cuda.is_bf16_supported():
            names.append("bfloat16")
        return tuple(names)
    if selection == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("this GPU/runtime does not report bfloat16 support")
    return (selection,)


def _torch_dtype(name: str) -> torch.dtype:
    return torch.bfloat16 if name == "bfloat16" else torch.float16


def _tolerances(name: str) -> Dict[str, float]:
    if name == "bfloat16":
        return {"atol": 2.0e-2, "rtol": 2.0e-2, "lse_atol": 5.0e-2}
    return {"atol": 6.0e-3, "rtol": 6.0e-3, "lse_atol": 2.0e-2}


def _latency_stats(samples_ms: List[float]) -> Dict[str, Any]:
    ordered = sorted(samples_ms)
    p95_index = min(len(ordered) - 1, max(0, int(0.95 * len(ordered)) - 1))
    return {
        "samples_ms": samples_ms,
        "mean_ms": statistics.mean(samples_ms),
        "median_ms": statistics.median(samples_ms),
        "p95_ms": ordered[p95_index],
        "std_ms": statistics.pstdev(samples_ms),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def _benchmark_cuda(
    function: Callable[[], Any], warmup: int, iterations: int
) -> Dict[str, Any]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()

    starts: List[torch.cuda.Event] = []
    ends: List[torch.cuda.Event] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        starts.append(start)
        ends.append(end)
    torch.cuda.synchronize()
    samples = [start.elapsed_time(end) for start, end in zip(starts, ends)]
    return _latency_stats(samples)


def _error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> Dict[str, float]:
    delta = actual.float() - expected.float()
    absolute = delta.abs()
    return {
        "max_abs": absolute.max().item(),
        "mean_abs": absolute.mean().item(),
        "rmse": delta.square().mean().sqrt().item(),
    }


def _prefill_result(
    case: EvidenceCase,
    dtype_name: str,
    warmup: int,
    iterations: int,
) -> Dict[str, Any]:
    dtype = _torch_dtype(dtype_name)
    tolerances = _tolerances(dtype_name)
    B = case.batch_size
    Hq = case.num_heads
    Hkv = case.num_kv_heads
    T = case.context_length
    D = case.head_dim

    q = torch.randn(B, Hq, T, D, device="cuda", dtype=dtype) * 0.1
    k = torch.randn(B, Hkv, T, D, device="cuda", dtype=dtype) * 0.1
    v = torch.randn(B, Hkv, T, D, device="cuda", dtype=dtype) * 0.1
    cos, sin = precompute_rope_cos_sin(T, D, "cuda", dtype)

    group_size = Hq // Hkv
    k_expanded = k.repeat_interleave(group_size, dim=1)
    v_expanded = v.repeat_interleave(group_size, dim=1)

    q_reference = apply_rope_reference(q.float(), cos.float(), sin.float())
    k_reference = apply_rope_reference(k_expanded.float(), cos.float(), sin.float())
    output_reference = F.scaled_dot_product_attention(
        q_reference, k_reference, v_expanded.float(), is_causal=True
    )

    output_actual, lse_actual = dominus_ultra_prefill(
        q, k, v, cos, sin, num_kv_heads=Hkv
    )
    output_errors = _error_metrics(output_actual, output_reference)
    output_pass = torch.allclose(
        output_actual.float(),
        output_reference,
        atol=tolerances["atol"],
        rtol=tolerances["rtol"],
    )

    scores = torch.matmul(q_reference, k_reference.transpose(-2, -1)) / (D**0.5)
    causal_mask = torch.ones((T, T), device="cuda", dtype=torch.bool).triu(1)
    lse_reference = torch.logsumexp(
        scores.masked_fill(causal_mask, -float("inf")), dim=-1
    )
    lse_errors = _error_metrics(lse_actual, lse_reference)
    lse_pass = torch.allclose(
        lse_actual,
        lse_reference,
        atol=tolerances["lse_atol"],
        rtol=tolerances["rtol"],
    )

    q_timed = apply_rope_reference(q, cos, sin)
    k_timed = apply_rope_reference(k_expanded, cos, sin)

    def run_kernel() -> Any:
        return dominus_ultra_prefill(q, k, v, cos, sin, num_kv_heads=Hkv)

    def run_baseline() -> torch.Tensor:
        return F.scaled_dot_product_attention(
            q_timed, k_timed, v_expanded, is_causal=True
        )

    kernel_timing = _benchmark_cuda(run_kernel, warmup, iterations)
    baseline_timing = _benchmark_cuda(run_baseline, warmup, iterations)
    speedup = baseline_timing["median_ms"] / kernel_timing["median_ms"]
    token_positions = B * T

    return {
        "case": asdict(case),
        "label": case.label,
        "dtype": dtype_name,
        "status": "pass" if output_pass and lse_pass else "fail",
        "correctness": {
            "output_pass": output_pass,
            "lse_pass": lse_pass,
            "output_error": output_errors,
            "lse_error": lse_errors,
            "tolerances": tolerances,
        },
        "timing_method": "CUDA events; compile/autotune excluded by warmup",
        "baseline": "PyTorch SDPA with RoPE precomputed outside the timed region",
        "kernel": kernel_timing,
        "baseline_timing": baseline_timing,
        "speedup_median": speedup,
        "kernel_token_positions_per_second": token_positions
        / (kernel_timing["median_ms"] / 1000.0),
        "baseline_token_positions_per_second": token_positions
        / (baseline_timing["median_ms"] / 1000.0),
        "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
    }


def _decode_result(
    case: EvidenceCase,
    dtype_name: str,
    warmup: int,
    iterations: int,
) -> Dict[str, Any]:
    dtype = _torch_dtype(dtype_name)
    tolerances = _tolerances(dtype_name)
    B = case.batch_size
    Hq = case.num_heads
    Hkv = case.num_kv_heads
    T = case.context_length
    D = case.head_dim

    q = torch.randn(B, Hq, 1, D, device="cuda", dtype=dtype) * 0.1
    k = torch.randn(B, Hkv, T, D, device="cuda", dtype=dtype) * 0.1
    v = torch.randn(B, Hkv, T, D, device="cuda", dtype=dtype) * 0.1
    cos, sin = precompute_rope_cos_sin(T + 1, D, "cuda", dtype)

    group_size = Hq // Hkv
    k_expanded = k.repeat_interleave(group_size, dim=1)
    v_expanded = v.repeat_interleave(group_size, dim=1)
    q_reference = apply_rope_reference(
        q.float(), cos[T : T + 1].float(), sin[T : T + 1].float()
    )
    k_reference = apply_rope_reference(
        k_expanded.float(), cos[:T].float(), sin[:T].float()
    )
    output_reference = F.scaled_dot_product_attention(
        q_reference, k_reference, v_expanded.float(), is_causal=False
    )
    output_actual = dominus_ultra_decode(q, k, v, cos, sin, num_kv_heads=Hkv)
    output_errors = _error_metrics(output_actual, output_reference)
    output_pass = torch.allclose(
        output_actual.float(),
        output_reference,
        atol=tolerances["atol"],
        rtol=tolerances["rtol"],
    )

    q_timed = apply_rope_reference(q, cos[T : T + 1], sin[T : T + 1])
    k_timed = apply_rope_reference(k_expanded, cos[:T], sin[:T])

    def run_kernel() -> torch.Tensor:
        return dominus_ultra_decode(q, k, v, cos, sin, num_kv_heads=Hkv)

    def run_baseline() -> torch.Tensor:
        return F.scaled_dot_product_attention(
            q_timed, k_timed, v_expanded, is_causal=False
        )

    kernel_timing = _benchmark_cuda(run_kernel, warmup, iterations)
    baseline_timing = _benchmark_cuda(run_baseline, warmup, iterations)
    speedup = baseline_timing["median_ms"] / kernel_timing["median_ms"]

    return {
        "case": asdict(case),
        "label": case.label,
        "dtype": dtype_name,
        "status": "pass" if output_pass else "fail",
        "correctness": {
            "output_pass": output_pass,
            "output_error": output_errors,
            "tolerances": tolerances,
        },
        "timing_method": "CUDA events; compile/autotune excluded by warmup",
        "baseline": "PyTorch SDPA with RoPE precomputed outside the timed region",
        "kernel": kernel_timing,
        "baseline_timing": baseline_timing,
        "speedup_median": speedup,
        "kernel_token_positions_per_second": B / (kernel_timing["median_ms"] / 1000.0),
        "baseline_token_positions_per_second": B
        / (baseline_timing["median_ms"] / 1000.0),
        "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
    }


def _run_case(
    case: EvidenceCase,
    dtype_name: str,
    warmup: int,
    iterations: int,
    seed: int,
) -> Dict[str, Any]:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        if case.mode == "prefill":
            return _prefill_result(case, dtype_name, warmup, iterations)
        return _decode_result(case, dtype_name, warmup, iterations)
    except Exception as error:  # Preserve compile/runtime failures as evidence.
        return {
            "case": asdict(case),
            "label": case.label,
            "dtype": dtype_name,
            "status": "error",
            "error_type": type(error).__name__,
            "error": str(error)[:4000],
        }
    finally:
        gc.collect()
        torch.cuda.empty_cache()


def _markdown(report: Dict[str, Any]) -> str:
    payload = report["payload"]
    environment = payload["environment"]
    results = payload["results"]
    lines = [
        "# DominusUltra GPU evidence report",
        "",
        f"**Verdict:** `{payload['verdict'].upper()}`  ",
        f"**Payload SHA-256:** `{report['payload_sha256']}`  ",
        f"**Generated:** {environment['generated_at']}  ",
        f"**Commit:** `{environment['git']['commit']}`  ",
        f"**Dirty worktree:** `{environment['git']['dirty']}`",
        "",
        "## Environment",
        "",
        f"- GPU: {environment['device_name']}",
        f"- Compute capability: {environment['compute_capability']}",
        f"- CUDA runtime: {environment['cuda_runtime']}",
        f"- PyTorch: {environment['torch']}",
        f"- Triton: {environment['triton']}",
        f"- Python: {environment['python']}",
        f"- Command: `{environment['command']}`",
        "",
        "## Cases",
        "",
        "| Case | Dtype | Status | Max error | Kernel median | Baseline median | Speedup |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        correctness = result.get("correctness", {})
        max_error = correctness.get("output_error", {}).get("max_abs")
        kernel_ms = result.get("kernel", {}).get("median_ms")
        baseline_ms = result.get("baseline_timing", {}).get("median_ms")
        speedup = result.get("speedup_median")
        lines.append(
            "| {label} | {dtype} | {status} | {error} | {kernel} | {baseline} | {speedup} |".format(
                label=result["label"],
                dtype=result["dtype"],
                status=result["status"],
                error=f"{max_error:.6g}" if max_error is not None else "—",
                kernel=f"{kernel_ms:.4f} ms" if kernel_ms is not None else "—",
                baseline=f"{baseline_ms:.4f} ms" if baseline_ms is not None else "—",
                speedup=f"{speedup:.3f}x" if speedup is not None else "—",
            )
        )

    failures = [result for result in results if result["status"] != "pass"]
    if failures:
        lines.extend(("", "## Failures", ""))
        for failure in failures:
            detail = failure.get("error", "correctness tolerance exceeded")
            lines.append(f"- `{failure['label']}` ({failure['dtype']}): {detail}")

    lines.extend(
        (
            "",
            "## Interpretation boundary",
            "",
            "- Throughput means token positions processed by this isolated kernel call; it is not end-to-end model generation speed.",
            "- The baseline is PyTorch SDPA with RoPE precomputed outside its timed region, a conservative comparison for a fused-RoPE kernel.",
            "- Raw CUDA-event samples and correctness metrics are preserved in the companion JSON file.",
            "- A PASS applies only to the recorded commit, hardware, software stack, shapes, dtypes, and tolerances.",
            "",
        )
    )
    return "\n".join(lines)


def _write_report(output_dir: Path, payload: Dict[str, Any]) -> Tuple[Path, Path, str]:
    canonical_payload = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    payload_sha256 = hashlib.sha256(canonical_payload).hexdigest()
    report = {
        "schema_version": SCHEMA_VERSION,
        "payload_sha256": payload_sha256,
        "payload": payload,
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"dominus-ultra-evidence-{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{stem}.json"
    markdown_path = output_dir / f"{stem}.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    return json_path, markdown_path, payload_sha256


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run correctness-gated DominusUltra GPU evidence cases"
    )
    parser.add_argument("--suite", choices=("quick", "full"), default="quick")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "both"),
        default="auto",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results"))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.warmup < 1 or args.iterations < 2:
        print("--warmup must be >= 1 and --iterations must be >= 2", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print(
            "CUDA is required; torch.cuda.is_available() returned false",
            file=sys.stderr,
        )
        return 2

    command = shlex.join((sys.executable, *sys.argv))
    try:
        dtypes = _dtype_names(args.dtype)
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 2

    cases = QUICK_CASES if args.suite == "quick" else FULL_CASES
    results: List[Dict[str, Any]] = []
    print(f"DominusUltra evidence: {len(cases)} cases x {len(dtypes)} dtype(s)")
    for dtype_index, dtype_name in enumerate(dtypes):
        for case_index, case in enumerate(cases):
            case_seed = args.seed + dtype_index * 10_000 + case_index
            print(
                f"[{len(results) + 1}/{len(cases) * len(dtypes)}] {case.label} {dtype_name}"
            )
            result = _run_case(
                case,
                dtype_name,
                args.warmup,
                args.iterations,
                case_seed,
            )
            results.append(result)
            print(f"  -> {result['status'].upper()}")

    verdict = (
        "pass" if all(result["status"] == "pass" for result in results) else "fail"
    )
    payload = {
        "verdict": verdict,
        "suite": args.suite,
        "dtypes": list(dtypes),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "seed": args.seed,
        "environment": _environment_metadata(command),
        "results": results,
    }
    json_path, markdown_path, payload_sha256 = _write_report(args.output_dir, payload)
    print(f"Verdict: {verdict.upper()}")
    print(f"Payload SHA-256: {payload_sha256}")
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")
    return 0 if verdict == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
