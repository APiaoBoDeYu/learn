"""统一的计时与对比工具。

核心原则（第 00 章详细讲解）：
1. CUDA 是异步的：Python 层的函数返回时，GPU 可能还没算完，计时前后都要同步。
2. 必须 warmup：首次调用会触发 cuDNN/cuBLAS 选算法、CUDA context 初始化、torch.compile 编译等一次性开销。
3. 看分布，不只看均值：报告 p50/p90/p99，延迟敏感场景主要看尾延迟。
"""
from __future__ import annotations

import contextlib
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable

import torch


@dataclass
class Stats:
    name: str
    times_ms: list[float] = field(repr=False)
    batch: int = 1

    def _pct(self, q: float) -> float:
        xs = sorted(self.times_ms)
        k = (len(xs) - 1) * q
        lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
        return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)

    @property
    def mean(self) -> float: return statistics.fmean(self.times_ms)
    @property
    def std(self) -> float: return statistics.pstdev(self.times_ms)
    @property
    def min(self) -> float: return min(self.times_ms)
    @property
    def p50(self) -> float: return self._pct(0.50)
    @property
    def p90(self) -> float: return self._pct(0.90)
    @property
    def p99(self) -> float: return self._pct(0.99)
    @property
    def throughput(self) -> float:
        """每秒处理样本数（按 p50 计算）。"""
        return self.batch * 1000.0 / self.p50

    def summary(self) -> str:
        return (f"{self.name:<28s} p50={self.p50:9.3f}ms  p90={self.p90:9.3f}ms  "
                f"p99={self.p99:9.3f}ms  mean={self.mean:9.3f}ms  std={self.std:7.3f}  "
                f"thr={self.throughput:10.1f}/s")


def benchmark(fn: Callable[[], object], *, name: str = "fn", warmup: int = 10, iters: int = 50,
              mode: str = "wall", batch: int = 1) -> Stats:
    """测量 fn() 的单次耗时。

    mode="wall"  : perf_counter + 前后 synchronize。测的是端到端延迟（包含 CPU 开销），适合流水线。
    mode="event" : CUDA Event 在 GPU 时间线上打点。测的是两个 event 之间的 GPU 时间，适合纯 GPU 算子。
                   注意：如果 fn 内部有 CPU 阻塞，event 测到的时间也会包含 GPU 空等的那段。
    """
    use_cuda = torch.cuda.is_available()
    for _ in range(warmup):
        fn()
    if use_cuda:
        torch.cuda.synchronize()

    times: list[float] = []
    if mode == "event" and use_cuda:
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            starts[i].record()
            fn()
            ends[i].record()
        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    elif mode == "wall":
        for _ in range(iters):
            if use_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            if use_cuda:
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return Stats(name=name, times_ms=times, batch=batch)


def compare(results: list[Stats], baseline: int = 0) -> str:
    """打印对比表，以 results[baseline] 为基准计算加速比（按 p50）。"""
    base = results[baseline].p50
    lines = [f"{'name':<28s} {'p50(ms)':>10s} {'p90(ms)':>10s} {'p99(ms)':>10s} {'speedup':>8s}",
             "-" * 70]
    for r in results:
        lines.append(f"{r.name:<28s} {r.p50:10.3f} {r.p90:10.3f} {r.p99:10.3f} {base / r.p50:7.2f}x")
    text = "\n".join(lines)
    print(text)
    return text


@contextlib.contextmanager
def cuda_timer(name: str = "region", verbose: bool = True):
    """一次性的计时上下文：with cuda_timer("xx") as t: ...; 之后读取 t["ms"]。"""
    out: dict[str, float] = {}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        yield out
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        out["ms"] = (time.perf_counter() - t0) * 1000.0
        if verbose:
            print(f"[timer] {name}: {out['ms']:.3f} ms")


@contextlib.contextmanager
def region(name: str):
    """统一的区域标注：同时生成 NVTX range（nsys 可见）和 record_function（torch.profiler 可见）。

    没有开启 profiler 时开销在微秒级，可以一直保留在代码里。
    """
    with torch.profiler.record_function(name):
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
