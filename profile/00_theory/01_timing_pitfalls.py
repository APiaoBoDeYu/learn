"""00-01 计时的坑：不同步 / 不 warmup / 只看均值。

运行：/opt/venv/bin/python 00_theory/01_timing_pitfalls.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common import TinyCNN, benchmark, print_env  # noqa: E402


def wall(fn, sync: bool) -> float:
    if sync:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    if sync:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000


def main():
    print_env()
    a = torch.randn(4096, 4096, device="cuda")
    b = torch.randn(4096, 4096, device="cuda")
    mm = lambda: a @ b  # noqa: E731

    print("\n=== 坑 1：首次调用（冷启动）===")
    # 第一次调用 matmul 会初始化 cuBLAS handle、选择 kernel 等
    print(f"第 1 次调用: {wall(mm, sync=True):8.3f} ms")
    print(f"第 2 次调用: {wall(mm, sync=True):8.3f} ms")
    print(f"第 3 次调用: {wall(mm, sync=True):8.3f} ms")

    print("\n=== 坑 2：不同步就计时 ===")
    for _ in range(5):
        mm()
    torch.cuda.synchronize()
    no_sync = [wall(mm, sync=False) for _ in range(20)]
    torch.cuda.synchronize()
    with_sync = [wall(mm, sync=True) for _ in range(20)]
    print(f"不 sync:  median {sorted(no_sync)[10]:8.3f} ms  <- 只测到了 kernel launch 的 CPU 耗时")
    print(f"有 sync:  median {sorted(with_sync)[10]:8.3f} ms  <- 真实的 GPU 执行时间")

    # 更隐蔽的情况：循环里不同步，队列满了以后 CPU 会被反压，单次测量值忽大忽小
    torch.cuda.synchronize()
    seq = [wall(mm, sync=False) for _ in range(2000)]
    torch.cuda.synchronize()
    print(f"连续 2000 次不 sync: 前 5 次 {[round(x, 3) for x in seq[:5]]}")
    print(f"                     后 5 次 {[round(x, 3) for x in seq[-5:]]}  <- launch 队列满后 CPU 被阻塞")

    print("\n=== 坑 3：wall-clock vs CUDA Event ===")
    s1 = benchmark(mm, name="matmul wall", mode="wall")
    s2 = benchmark(mm, name="matmul event", mode="event")
    print(s1.summary())
    print(s2.summary())
    flops = 2 * 4096 ** 3
    print(f"=> FP32 matmul 实测算力 ≈ {flops / s2.p50 / 1e9:.2f} TFLOPS")

    print("\n=== 坑 4：小 kernel 时 wall 比 event 多出 launch/同步开销 ===")
    x = torch.randn(1024, device="cuda")
    small = lambda: x * 2  # noqa: E731
    print(benchmark(small, name="tiny op wall", mode="wall", iters=200).summary())
    print(benchmark(small, name="tiny op event", mode="event", iters=200).summary())

    print("\n=== 坑 5：只看均值会掩盖抖动 ===")
    model = TinyCNN().cuda().eval()
    inp = torch.randn(8, 3, 224, 224, device="cuda")
    k = [0]

    @torch.no_grad()
    def jittery():
        k[0] += 1
        if k[0] % 10 == 0:          # 每 10 次插入一次 CPU 卡顿（模拟 GC、日志刷盘、其他进程抢占）
            time.sleep(0.02)
        model(inp)

    s = benchmark(jittery, name="cnn with hiccup", iters=100)
    print(s.summary())
    print(f"mean/p50 = {s.mean / s.p50:.2f}，p99/p50 = {s.p99 / s.p50:.2f}  <- 尾延迟才是实时系统关心的")


if __name__ == "__main__":
    main()
