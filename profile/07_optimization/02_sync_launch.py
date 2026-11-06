"""07-02 同步点与 Launch-bound，以及这些手段在什么情况下无效

    /opt/venv/bin/python 07_optimization/02_sync_launch.py

A. 同步点
   A1 一次 GEMM（约 2 ms）+ CPU 上 2 ms 的工作：中间有没有 .item()，决定了 CPU 和 GPU 能否并行
   A2 流水线的后处理（common/pipeline.py）：eager_small_ops × sync_in_loop 四种组合
B. Launch-bound：ManySmallOps(200)（200 个很小的逐元素算子，eager 下 300 个 kernel）
   eager / torch.compile / compile reduce-overhead / 手动 CUDA Graph
C. 无效的情况：同样的手段用在 Memory-bound（64M 元素的逐元素链）和 Compute-bound（GEMM 4096 fp16）上
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ManySmallOps, PipelineConfig, SlowPipeline, benchmark, compare  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs"
lines = []


def p(s=""):
    print(s); lines.append(s)


def cpu_work(ms=2.0):
    t = time.perf_counter()
    while (time.perf_counter() - t) * 1000 < ms:
        pass


def graphed(fn, x):
    """手动 CUDA Graph：在旁路 stream 上 warmup，再 capture（第 04 章 4.6）。"""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn(x)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = fn(x)
    return lambda: (g.replay(), y)[1]


def part_a():
    p("== A1. GEMM（GPU）+ 2 ms CPU 工作 ==")
    a = torch.randn(4096, 4096, device="cuda")
    s = torch.zeros((), device="cuda")

    def gemm():
        torch.matmul(a, a, out=b)
    b = torch.empty_like(a)
    t_gemm = benchmark(gemm, name="只有 GEMM", mode="event", iters=20).p50

    def with_item():
        gemm()
        _ = b[0, 0].item()           # 同步：CPU 等 GEMM 做完
        cpu_work()

    def without_item():
        gemm()
        cpu_work()                   # CPU 工作和 GEMM 并行
        _ = b[0, 0].item()           # 真正需要结果时再同步
    r = [benchmark(with_item, name="GEMM → .item() → CPU 工作", iters=30),
         benchmark(without_item, name="GEMM → CPU 工作 → .item()", iters=30)]
    lines.append(compare(r))
    p(f"  GEMM 单独 {t_gemm:.3f} ms，CPU 工作 2.000 ms：串行之和 {t_gemm + 2:.3f} ms，完全并行的下限 {max(t_gemm, 2):.3f} ms")

    p("\n== A2. 流水线后处理（输入是 ToyVLA 的输出 (1,16,7)）==")
    raw = torch.rand(1, 16, 7, device="cuda") * 2 - 1
    r = []
    for eager in (True, False):
        for sync in (True, False):
            pipe = SlowPipeline(PipelineConfig.fast(eager_small_ops=eager, sync_in_loop=sync))
            r.append(benchmark(lambda: pipe.postprocess(raw), name=f"eager_small_ops={eager!s:<5s} sync={sync!s:<5s}", iters=100))
    lines.append(compare(r))


def part_b():
    p("\n== B. Launch-bound：ManySmallOps(200)，输入 1024 个元素 ==")
    m = ManySmallOps(200)
    x = torch.randn(1024, device="cuda")
    torch._dynamo.reset()
    c = torch.compile(m)
    cro = torch.compile(m, mode="reduce-overhead")
    with torch.no_grad():
        for f in (c, cro):
            for _ in range(3):
                f(x)
        g = graphed(m, x)
        ref = m(x)
        assert torch.allclose(g(), ref) and torch.allclose(c(x), ref, atol=1e-5)
        r = [benchmark(lambda: m(x), name="eager（300 个 kernel）", mode="event"),
             benchmark(g, name="手动 CUDA Graph", mode="event"),
             benchmark(lambda: c(x), name="torch.compile（融合）", mode="event"),
             benchmark(lambda: cro(x), name="compile reduce-overhead", mode="event")]
    lines.append(compare(r))


def part_c():
    p("\n== C1. 同样的手段用在 Memory-bound 上：x 有 64M 个元素（256 MB），10 个逐元素算子（x*a+b 在 eager 下是 mul、add 两个 kernel，共 15 个 kernel）==")
    m = ManySmallOps(10)
    x = torch.randn(64 * 1024 * 1024, device="cuda")
    torch._dynamo.reset()
    c = torch.compile(m)
    with torch.no_grad():
        for _ in range(3):
            c(x)
        g = graphed(m, x)
        r = [benchmark(lambda: m(x), name="eager（15 个 kernel）", mode="event", iters=20),
             benchmark(g, name="手动 CUDA Graph", mode="event", iters=20),
             benchmark(lambda: c(x), name="torch.compile（融合）", mode="event", iters=20)]
    lines.append(compare(r))
    p("  → CUDA Graph 只省 launch 开销，对 Memory-bound 无效；融合把 15 次读写 256 MB 变成 1 次，所以有效")

    p("\n== C2. Compute-bound：GEMM 4096 fp16 ==")
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.half)
    f = lambda t: t @ t  # noqa: E731
    torch._dynamo.reset()
    c = torch.compile(f)
    for _ in range(3):
        c(a)
    g = graphed(f, a)
    r = [benchmark(lambda: f(a), name="eager", mode="event", iters=30),
         benchmark(g, name="手动 CUDA Graph", mode="event", iters=30),
         benchmark(lambda: c(a), name="torch.compile", mode="event", iters=30)]
    lines.append(compare(r))
    p("  → GEMM 本身已经是 cuBLAS 的高效 kernel，launch 开销相对 ms 级的计算可以忽略")


if __name__ == "__main__":
    part_a(); part_b(); part_c()
    (OUT / "02_sync_launch.log").write_text("\n".join(lines) + "\n")
