"""第 05 章的自定义 CUDA kernel：用 nvcc 编译成 .so，再用 ctypes 调用。

为什么不用 torch.utils.cpp_extension.load_inline？
    load_inline 要编译 torch 的头文件，并且按 TORCH_CUDA_ARCH_LIST 生成多个架构（本机默认 6 个），
    实测第一次编译要 102 s。这里的 kernel 只用到裸指针，直接 nvcc 编译只需几秒，
    而且编译选项（-maxrregcount、--resource-usage、-lineinfo）一目了然。两种方式在 ncu 中看到的 kernel 没有区别。

用法：
    from kernels import K
    K.scale_coalesced(x, y, 2.0)        # x、y 是 CUDA 上的 float32 张量
"""
from __future__ import annotations

import ctypes
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
SRC = HERE / "cuda" / "bottlenecks.cu"
BUILD = HERE / "build"
NVCC = "/usr/local/cuda/bin/nvcc"
ARCH = "sm_110"   # Thor


def build(extra_flags: list[str] | None = None, tag: str = "default", verbose: bool = False) -> Path:
    """编译 bottlenecks.cu → build/libbottlenecks_<tag>.so；源文件没变就直接复用。"""
    BUILD.mkdir(exist_ok=True)
    so = BUILD / f"libbottlenecks_{tag}.so"
    if so.exists() and so.stat().st_mtime > SRC.stat().st_mtime:
        return so
    cmd = [NVCC, "-O3", "-lineinfo", f"-arch={ARCH}", "--shared", "-Xcompiler", "-fPIC",
           *(extra_flags or []), str(SRC), "-o", str(so)]
    if verbose:
        print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return so


def resource_usage(extra_flags: list[str] | None = None) -> str:
    """nvcc --resource-usage：编译期就能看到每个 kernel 的寄存器数、spill、shared memory 用量。"""
    cmd = [NVCC, "-O3", f"-arch={ARCH}", "--resource-usage", "-c", str(SRC), "-o", "/dev/null", *(extra_flags or [])]
    r = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return r.stdout + r.stderr


def resource_table(extra_flags: list[str] | None = None) -> dict[str, dict]:
    """解析 --resource-usage 的输出：{kernel 名: {regs, spill_st, spill_ld, stack, smem}}。"""
    import re
    res, cur = {}, None
    for line in resource_usage(extra_flags).splitlines():
        m = re.search(r"Compiling entry function '(\w+)'", line)
        if m:
            name = subprocess.run(["c++filt", m.group(1)], capture_output=True, text=True).stdout.strip()
            cur = name.split("(")[0].replace("void ", "")   # 例如 filter128_per_thread<4>
            res[cur] = {"regs": 0, "spill_st": 0, "spill_ld": 0, "stack": 0, "smem": 0}
            continue
        if cur is None:
            continue
        for key, pat in (("stack", r"(\d+) bytes stack frame"), ("spill_st", r"(\d+) bytes spill stores"),
                         ("spill_ld", r"(\d+) bytes spill loads"), ("regs", r"Used (\d+) registers"),
                         ("smem", r"(\d+) bytes smem")):
            m = re.search(pat, line)
            if m:
                res[cur][key] = int(m.group(1))
    return res


def _p(t: torch.Tensor) -> ctypes.c_void_p:
    assert t.is_cuda and t.is_contiguous()
    return ctypes.c_void_p(t.data_ptr())


def _s() -> ctypes.c_void_p:
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


class _Kernels:
    def __init__(self, so: Path):
        self.lib = ctypes.CDLL(str(so))
        i64, i32, f32 = ctypes.c_int64, ctypes.c_int, ctypes.c_float
        self._i64, self._i32, self._f32 = i64, i32, f32

    # ---- 1. memory-bound ----
    def scale_coalesced(self, x, y, a):
        self.lib.launch_scale_coalesced(_p(x), _p(y), self._f32(a), self._i64(x.numel()), _s())

    def scale_strided(self, x, y, a, stride=32):
        self.lib.launch_scale_strided(_p(x), _p(y), self._f32(a), self._i64(x.numel()), self._i32(stride), _s())

    def scale_vec4(self, x, y, a):
        assert x.numel() % 4 == 0
        self.lib.launch_scale_vec4(_p(x), _p(y), self._f32(a), self._i64(x.numel()), _s())

    def fused_axpb_relu(self, x, y, a, b):
        self.lib.launch_fused_axpb_relu(_p(x), _p(y), self._f32(a), self._f32(b), self._i64(x.numel()), _s())

    # ---- 2. register-bound ----  variant: 0=每线程一个矩阵 1=同上+__maxnreg__(64) 2=每线程一行
    def bmm8(self, variant, A, B, C, block=128):
        batch = A.shape[1]
        self.lib.launch_bmm8(self._i32(variant), _p(A), _p(B), _p(C), self._i64(batch), self._i32(block), _s())

    def filter128(self, variant, x, out):  # variant: 0=每线程 128 个状态 1=4 个线程分担 2=0+预取 3=1+预取
        T, n = x.shape
        self.lib.launch_filter(self._i32(variant), _p(x), _p(out), self._i32(T), self._i64(n), _s())

    # ---- 3. control-flow ----  variant: 0=按线程奇偶分支（divergent） 1=按 warp 分支
    def branch(self, variant, x, iters):
        self.lib.launch_branch(self._i32(variant), _p(x), self._i64(x.numel()), self._i32(iters), _s())

    def loop_varcount(self, x, cnt):
        assert cnt.dtype == torch.int32
        self.lib.launch_loop_varcount(_p(x), _p(cnt), self._i64(x.numel()), _s())

    # ---- 4. bank conflict ----  variant: 0=naive 1=smem 2=smem+padding 3=smem+swizzle 4=纯拷贝
    def transpose(self, variant, inp, out):
        rows, cols = inp.shape
        assert rows % 32 == 0 and cols % 32 == 0
        self.lib.launch_transpose(self._i32(variant), _p(inp), _p(out), self._i32(rows), self._i32(cols), _s())


_cache: dict[str, _Kernels] = {}


def load(extra_flags: list[str] | None = None, tag: str = "default") -> _Kernels:
    if tag not in _cache:
        _cache[tag] = _Kernels(build(extra_flags, tag))
    return _cache[tag]


class _Lazy:
    def __getattr__(self, name):
        return getattr(load(), name)


K = _Lazy()

if __name__ == "__main__":
    print(build(verbose=True))
    print(resource_usage(sys.argv[1:]))
