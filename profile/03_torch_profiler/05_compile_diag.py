"""03-05 torch.compile 诊断：graph break、重编译、mode 对比。

A. graph break：慢/快两版后处理各有多少 graph break，原因是什么（explain + counters）
B. 编译效果：eager vs compile(default) vs compile(reduce-overhead = CUDA Graph)
C. 重编译：输入 shape 变化导致重新编译；用 dynamic=True / mark_dynamic 解决

命令行诊断（不改代码）：
    TORCH_LOGS="graph_breaks,recompiles" python xxx.py
    TORCH_LOGS="+dynamo" python xxx.py        # 非常详细
    TORCH_COMPILE_DEBUG=1 python xxx.py       # 把生成的代码保存到 torch_compile_debug/

运行：/opt/venv/bin/python 03_torch_profiler/05_compile_diag.py   （首次编译需要 1~2 分钟）
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402  （common 已设置 TRITON_PTXAS_PATH）
import torch._dynamo as dynamo  # noqa: E402

from common import ManySmallOps, ToyVLA, benchmark, compare  # noqa: E402

SCALE = torch.tensor([0.05, 0.05, 0.05, 0.2, 0.2, 0.2, 1.0], device="cuda")


def post_slow(a, prev):
    """慢流水线后处理的纯张量部分：按维度、按时间步循环，并用 .item() 检查越界。"""
    out = torch.empty_like(a)
    warn = 0
    for d in range(a.shape[1]):
        col = a[:, d] * SCALE[d]
        col = torch.clamp(col, -SCALE[d], SCALE[d])
        for t in range(a.shape[0]):
            col[t] = 0.8 * col[t] + 0.2 * prev[t, d]
        if col.abs().max().item() > 0.99 * SCALE[d].item():
            warn += 1
        out[:, d] = col
    return out, warn


def post_fast(a, prev):
    scaled = (a * SCALE).clamp(-SCALE, SCALE)
    return 0.8 * scaled + 0.2 * prev


def part_a():
    print("=== A. graph break 分析 ===")
    from torch._dynamo.utils import counters
    a, prev = torch.randn(16, 7, device="cuda"), torch.zeros(16, 7, device="cuda")
    for name, fn in [("post_slow", post_slow), ("post_fast", post_fast)]:
        # 方法 1：torch._dynamo.explain（遇到 .item() 这类 break 时，2.9 版本的统计可能不准）
        dynamo.reset()
        ex = dynamo.explain(fn)(a, prev)
        # 方法 2：实际编译运行一次，读取 dynamo 的计数器（更可靠）
        dynamo.reset()
        counters.clear()
        torch.compile(fn)(a, prev)
        breaks = dict(counters["graph_break"])
        unimpl = dict(counters["unimplemented"])
        print(f"\n[{name}] explain: graphs={ex.graph_count} breaks={ex.graph_break_count} | counters: "
              f"graph_break={sum(breaks.values())} unimplemented={sum(unimpl.values())} "
              f"编译出的图={counters['stats']['unique_graphs']}")
        for r, n in sorted({**breaks, **unimpl}.items(), key=lambda x: -x[1])[:4]:
            print(f"   {n:3d} x  {r.splitlines()[0][:110]}")
        if counters["stats"]["unique_graphs"] == 0:
            print("   → 一个图都没编译出来：graph break 发生在 for 循环内部时，dynamo 会放弃整个函数，退回 eager 执行"
                  "（不报错，非常隐蔽）")


def timed_compile(fn, *args, **kw):
    t0 = time.perf_counter()
    c = torch.compile(fn, **kw)
    c(*args)
    torch.cuda.synchronize()
    return c, time.perf_counter() - t0


def part_b():
    print("\n=== B. 编译效果 ===")
    torch._dynamo.config.cache_size_limit = 64
    cases = []
    x = torch.randn(1024, device="cuda")
    small = ManySmallOps(200).cuda()
    cases.append(("ManySmallOps(200), 1K elems", small, (x,)))
    a, prev = torch.randn(16, 7, device="cuda"), torch.zeros(16, 7, device="cuda")
    cases.append(("post_fast (16x7)", post_fast, (a, prev)))
    vla = ToyVLA().cuda().half().eval()
    img = torch.randn(1, 3, 224, 224, device="cuda", dtype=torch.half)
    tok = torch.randint(0, 32000, (1, 32), device="cuda")
    cases.append(("ToyVLA fp16 batch=1", vla, (img, tok)))

    for name, fn, args in cases:
        dynamo.reset()
        with torch.no_grad():
            res = [benchmark(lambda: fn(*args), name="eager", warmup=10, iters=100)]
            c1, t1 = timed_compile(fn, *args)
            res.append(benchmark(lambda: c1(*args), name=f"compile default ({t1:.0f}s)", warmup=10, iters=100))
            dynamo.reset()
            c2, t2 = timed_compile(fn, *args, mode="reduce-overhead")
            for _ in range(3):  # reduce-overhead 前几次调用会录制 CUDA Graph
                c2(*args)
            res.append(benchmark(lambda: c2(*args), name=f"reduce-overhead ({t2:.0f}s)", warmup=10, iters=100))
        print(f"\n[{name}]")
        compare(res)


def part_c():
    print("\n=== C. 重编译：shape 变化 ===")
    from torch._dynamo.utils import counters

    def f(x):
        return torch.nn.functional.gelu(x) * 2 + 1

    for label, kw in [("默认（静态 shape）", {}), ("dynamic=True", {"dynamic": True})]:
        dynamo.reset()
        counters.clear()
        cf = torch.compile(f, **kw)
        t0 = time.perf_counter()
        for n in [128, 256, 384, 512, 640, 768]:
            cf(torch.randn(n, 512, device="cuda"))
        torch.cuda.synchronize()
        frames = sum(counters["stats"].values()) and counters["stats"].get("unique_graphs", 0)
        print(f"{label:<20s} 6 种 batch 大小 → 编译出的图 {frames} 个，总耗时 {time.perf_counter() - t0:.1f}s")
    print("说明：默认模式下第 1 次按静态 shape 编译，第 2 次遇到新 shape 时自动转为 dynamic 重编译，之后不再重编译；"
          "\n      dynamic=True 一开始就按动态 shape 编译。用 TORCH_LOGS=recompiles 可以看到每次重编译的原因。")


if __name__ == "__main__":
    part_a()
    part_b()
    part_c()
