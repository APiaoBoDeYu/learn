"""Shared utilities for the Triton / torch.compile tutorial examples.

Import this module BEFORE importing `triton` in every example script — it patches
os.environ so Triton's nvidia backend can find a ptxas that knows this GPU's
compute capability, and so the C-extension build step can find cuda.h.
See 01_setup.md for why this is needed on this machine (Jetson Thor, sm_110a).
"""
import os

# --- Environment fixups for Jetson Thor / very new GPU architectures -------
# triton==3.6.0's bundled nvidia backend (ptxas-blackwell, CUDA 12.9) does not
# know sm_110a, AND torch 2.9.1's Inductor backend requires torch.library
# internals (specialize_impl) that only exist in triton<=3.5. So we pin to
# triton==3.5.0, whose single bundled ptxas is even older (CUDA 12.4) and
# also doesn't know sm_110a. Point it at the system CUDA 13.0 toolkit's ptxas
# via the documented env-var hook (TRITON_PTXAS_PATH, see triton/knobs.py).
os.environ.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda-13.0/bin/ptxas")

_cuda_include = "/usr/local/cuda-13.0/targets/sbsa-linux/include"
_cuda_lib = "/usr/local/cuda-13.0/targets/sbsa-linux/lib"
if _cuda_include not in os.environ.get("CPATH", ""):
    os.environ["CPATH"] = _cuda_include + os.pathsep + os.environ.get("CPATH", "")
if _cuda_lib not in os.environ.get("LIBRARY_PATH", ""):
    os.environ["LIBRARY_PATH"] = _cuda_lib + os.pathsep + os.environ.get("LIBRARY_PATH", "")

import time
import torch


def device_info() -> None:
    """Print basic environment info; call at the top of every example's __main__."""
    import triton
    print(f"torch      : {torch.__version__}")
    print(f"triton     : {triton.__version__}")
    print(f"cuda avail : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device     : {torch.cuda.get_device_name(0)}")
        print(f"capability : {torch.cuda.get_device_capability(0)}")
    print("-" * 60)


def check_close(actual: torch.Tensor, expected: torch.Tensor, name: str,
                 atol: float = 1e-2, rtol: float = 1e-2) -> bool:
    """Compare two tensors and print a PASS/FAIL line. Returns the bool result."""
    ok = torch.allclose(actual, expected, atol=atol, rtol=rtol)
    max_err = (actual - expected).abs().max().item()
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: allclose={ok}, max_abs_err={max_err:.3e}")
    return ok


def bench_ms(fn, warmup: int = 10, rep: int = 50) -> float:
    """Time a zero-arg callable on the GPU, returning median latency in milliseconds.

    Uses cuda events for accurate GPU timing (wall-clock time() would include
    Python dispatch overhead and any async queuing slack).
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
    return times[len(times) // 2]


def bench_cpu_ms(fn, warmup: int = 5, rep: int = 20) -> float:
    """Time a zero-arg callable including host-side overhead (compile/dispatch)."""
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





if __name__ == "__main__":
    device_info()

    import torch, triton, triton.language as tl

    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x + y, mask=mask)
    x = torch.rand(1024, device='cuda')
    y = torch.rand(1024, device='cuda')
    out = torch.empty_like(x)
    add_kernel[(triton.cdiv(1024, 256),)](x, y, out, 1024, BLOCK_SIZE=256)
    torch.cuda.synchronize()
    print(out)
    assert torch.allclose(out, x + y)

