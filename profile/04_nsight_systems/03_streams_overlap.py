"""04-03 Stream 与 overlap：拷贝和计算能否并行？多个小 kernel 能否并行？

A. H2D 拷贝 + 计算：单 stream 串行 vs 双 stream 流水线（pinned memory + non_blocking）
   A1 计算是访存密集型（类型转换）→ 与拷贝争抢 DRAM 带宽
   A2 计算是计算密集型（GEMM）  → 与拷贝可以重叠
B. 占不满 SM 的 kernel：单 stream vs 多 stream 并发
   B1 很短的 kernel（softmax，十几微秒）→ 瓶颈在 launch，多 stream 无益
   B2 很长但只用 2 个 block 的 kernel（Triton）→ 多 stream 可以把空闲的 SM 用起来
都用 NVTX 标注，可以用 nsys 查看时间线：
    nsys profile -t cuda,nvtx -o outputs/streams --force-overwrite true \\
        /opt/venv/bin/python 04_nsight_systems/03_streams_overlap.py

运行：/opt/venv/bin/python 04_nsight_systems/03_streams_overlap.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common import benchmark, compare, region  # noqa: E402

N_CHUNK = 8


def part_a():
    print("=== A. H2D 拷贝 + 计算（8 个 chunk，每个 32MB）===")
    host = [torch.randn(8 * 1024 * 1024).pin_memory() for _ in range(N_CHUNK)]   # 32 MB/块，pinned
    dev = [torch.empty(8 * 1024 * 1024, device="cuda") for _ in range(N_CHUNK)]
    w = torch.randn(2048, 2048, device="cuda", dtype=torch.half)
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.half)
    s_copy, s_comp = torch.cuda.Stream(), torch.cuda.Stream()

    computes = {
        "A1 访存型计算(转 fp16 + GEMM2048)": lambda i: dev[i][: 2048 * 2048].view(2048, 2048).half() @ w,
        "A2 计算型(GEMM4096)": lambda i: a @ a,
    }
    for title, compute in computes.items():
        def serial():
            for i in range(N_CHUNK):
                dev[i].copy_(host[i], non_blocking=True)
                compute(i)

        def pipelined():
            evs = []
            with torch.cuda.stream(s_copy):            # 拷贝全部放到 copy stream
                for i in range(N_CHUNK):
                    dev[i].copy_(host[i], non_blocking=True)
                    e = torch.cuda.Event()
                    e.record(s_copy)
                    evs.append(e)
            with torch.cuda.stream(s_comp):            # 计算 stream 等对应 chunk 的拷贝完成后再算
                for i in range(N_CHUNK):
                    s_comp.wait_event(evs[i])
                    compute(i)
            torch.cuda.current_stream().wait_stream(s_comp)

        def copy_only():
            for i in range(N_CHUNK):
                dev[i].copy_(host[i], non_blocking=True)

        def compute_only():
            for i in range(N_CHUNK):
                compute(i)

        print(f"\n[{title}]")
        with region(title):
            r = [benchmark(copy_only, name="只拷贝", mode="event"),
                 benchmark(compute_only, name="只计算", mode="event"),
                 benchmark(serial, name="串行（单 stream）", mode="event"),
                 benchmark(pipelined, name="流水线（双 stream）", mode="event")]
        compare(r, baseline=2)
        print(f"理论下限 max(拷贝, 计算) = {max(r[0].p50, r[1].p50):.3f} ms；"
              f"不重叠时 拷贝+计算 = {r[0].p50 + r[1].p50:.3f} ms")
    print(f"\nH2D 拷贝带宽 ≈ {N_CHUNK * 32 / r[0].p50 * 1000 / 1024:.1f} GB/s"
          "（pinned → device，Thor 上两端是同一块 DRAM；读写合计约为该值的 2 倍）")


def part_b():
    import triton
    import triton.language as tl

    @triton.jit
    def spin_kernel(x_ptr, n_iter, BLOCK: tl.constexpr):
        # 每个 program（block）对自己的一段数据反复做 FMA：计算密集、运行时间长，但只占少数 SM
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        v = tl.load(x_ptr + offs)
        for _ in range(n_iter):
            v = v * 0.999 + 0.001
        tl.store(x_ptr + offs, v)

    print("\n=== B. 占不满 SM 的 kernel：单 stream vs 8 个 stream ===")
    streams = [torch.cuda.Stream() for _ in range(8)]

    def run(fn_per_stream, use_streams):
        cur = torch.cuda.current_stream()
        for k in range(8):
            if use_streams:
                streams[k].wait_stream(cur)
                with torch.cuda.stream(streams[k]):
                    fn_per_stream(k)
            else:
                fn_per_stream(k)
        if use_streams:
            for st in streams:
                cur.wait_stream(st)

    xs = [torch.randn(4, 256, 1024, device="cuda") for _ in range(8)]

    def softmax_chain(k):
        x = xs[k]
        for _ in range(20):
            x = torch.softmax(x, dim=-1)

    ys = [torch.randn(2 * 1024, device="cuda") for _ in range(8)]

    def spin(k):
        spin_kernel[(2,)](ys[k], 20000, BLOCK=1024)   # grid=2：只用 2 个 block

    for title, fn in [("B1 短 kernel：20 次 softmax(4x256x1024)", softmax_chain),
                      ("B2 长 kernel：grid=2 的 Triton 循环 kernel", spin)]:
        print(f"\n[{title}]")
        with region(title):
            r = [benchmark(lambda: run(fn, False), name="单 stream", mode="event"),
                 benchmark(lambda: run(fn, True), name="8 个 stream", mode="event")]
        compare(r)


if __name__ == "__main__":
    part_a()
    part_b()
