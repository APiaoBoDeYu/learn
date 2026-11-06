"""Shared utilities for the torch->TensorRT route comparison tutorial.

Import this BEFORE torch_tensorrt / triton in every example script: it patches
os.environ so the TensorRT and Triton toolchains can find a ptxas that knows
this GPU's compute capability (Jetson Thor, sm_110a) and so any C-extension
build step can find cuda.h. See 01_setup.md for the full story.

Everything here is route-agnostic on purpose. Each route module (02_*.py ..
05_*.py) exposes a `build(model, example_inputs, **kw) -> callable` and the
benchmark driver in 06_benchmark_all.py treats them uniformly, so the
comparison is apples-to-apples: same model, same inputs, same timing code.
"""
import os

# --- Environment fixups for Jetson Thor / very new GPU architectures -------
# Same rationale as the Triton tutorial: the bundled ptxas in the wheels
# predates sm_110a, so point the toolchains at the system CUDA 13.0 one.
os.environ.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda-13.0/bin/ptxas")

_cuda_include = "/usr/local/cuda-13.0/targets/sbsa-linux/include"
_cuda_lib = "/usr/local/cuda-13.0/targets/sbsa-linux/lib"
if _cuda_include not in os.environ.get("CPATH", ""):
    os.environ["CPATH"] = _cuda_include + os.pathsep + os.environ.get("CPATH", "")
if _cuda_lib not in os.environ.get("LIBRARY_PATH", ""):
    os.environ["LIBRARY_PATH"] = _cuda_lib + os.pathsep + os.environ.get("LIBRARY_PATH", "")

# Torch-TensorRT and the ONNX route both print the same three "modelopt /
# TensorRT-LLM not installed" banners on import. They are harmless for this
# tutorial (we never compile a quantized or distributed model through them),
# and silencing them keeps the benchmark tables readable.
os.environ.setdefault("TORCH_TENSORRT_LOG_LEVEL", "ERROR")

import json
import statistics
import time
from pathlib import Path

import torch

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)


def device_info() -> None:
    """Print environment info; call at the top of every example's __main__."""
    print(f"torch      : {torch.__version__}")
    try:
        import tensorrt
        print(f"tensorrt   : {tensorrt.__version__}")
    except ImportError:
        print("tensorrt   : not installed")
    try:
        import torch_tensorrt
        print(f"torch_trt  : {torch_tensorrt.__version__}")
    except ImportError:
        print("torch_trt  : not installed")
    print(f"cuda avail : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device     : {torch.cuda.get_device_name(0)}")
        print(f"capability : {torch.cuda.get_device_capability(0)}")
    print("-" * 68)


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------
def bench_ms(fn, warmup: int = 20, rep: int = 100) -> dict:
    """Time a zero-arg callable on the GPU.

    Returns median / p90 / mean latency in ms. We report the median as the
    headline number (robust to the occasional scheduler hiccup) but keep p90
    because for a deployed inference service the tail matters more than the
    average. Uses cuda events so Python dispatch overhead and async queuing
    slack do not pollute the measurement.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(rep):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return {
        "median_ms": times[len(times) // 2],
        "p90_ms": times[int(len(times) * 0.9)],
        "mean_ms": statistics.fmean(times),
    }


def bench_wall_ms(fn, warmup: int = 5, rep: int = 20) -> float:
    """Median wall-clock latency in ms, including host-side overhead.

    Use this when the thing you care about *is* the host overhead — e.g.
    comparing eager Python dispatch against a route that runs the whole graph
    from one call. cuda events would hide exactly that difference.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(rep):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


# --------------------------------------------------------------------------
# Accuracy
# --------------------------------------------------------------------------
def compare(actual: torch.Tensor, expected: torch.Tensor, name: str,
            atol: float = 2e-2, rtol: float = 2e-2) -> dict:
    """Compare two tensors, print a PASS/FAIL line, return the metrics.

    For FP16/INT8 routes `allclose` alone is a bad signal: a single outlier
    element fails it while the model's actual predictions are unchanged. So we
    also report cosine similarity and, for classifier-shaped outputs, whether
    the argmax still agrees — that is what actually decides if a deployment is
    acceptable.
    """
    a = actual.detach().float().flatten()
    e = expected.detach().float().flatten()
    max_err = (a - e).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(a, e, dim=0).item()
    ok = torch.allclose(a, e, atol=atol, rtol=rtol)
    top1 = None
    if actual.dim() == 2:
        top1 = (actual.argmax(-1) == expected.argmax(-1)).float().mean().item()
    status = "PASS" if ok else "DIFF"
    extra = f", top1_agree={top1:.3f}" if top1 is not None else ""
    print(f"[{status}] {name}: max_abs_err={max_err:.3e}, cos_sim={cos:.6f}{extra}")
    return {"allclose": ok, "max_abs_err": max_err, "cos_sim": cos, "top1_agree": top1}


# --------------------------------------------------------------------------
# Result bookkeeping
# --------------------------------------------------------------------------
def save_result(tag: str, record: dict) -> None:
    """Append one result record to artifacts/results.jsonl.

    Each route script writes its own rows as it runs, and 06_benchmark_all.py
    reads the file back to render the comparison table. Keeping this as
    append-only JSONL means a route that crashes (or that you skip because a
    dependency is missing) does not invalidate the rows already collected.
    """
    record = {"tag": tag, "ts": time.time(), **record}
    with open(ARTIFACTS / "results.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")


def load_results() -> list:
    path = ARTIFACTS / "results.jsonl"
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def engine_size_mb(path) -> float:
    return Path(path).stat().st_size / 1024 / 1024


if __name__ == "__main__":
    device_info()
    x = torch.randn(4096, 4096, device="cuda", dtype=torch.half)
    print("matmul bench:", bench_ms(lambda: x @ x))
