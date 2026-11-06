"""04-04 CUDA Graph：消除 launch 开销。只对 Launch-bound 有效。

A. 三种负载：eager vs CUDA Graph
   - ManySmallOps(200)：极端 Launch-bound
   - ToyVLA fp16 batch=1：中度 Launch-bound（第 00 章测得 GPU busy 只有 54%）
   - 256MB 拷贝 × 4：Memory-bound（kernel 很少、很长）→ Graph 无效
B. capture 时的常见错误：.item() 同步、动态 shape、capture 期间分配显存
C. 用 nsys 验证：eager 每次迭代上百次 cudaLaunchKernel，Graph 只有 1 次 cudaGraphLaunch
    nsys profile -t cuda,nvtx -o outputs/graph --force-overwrite true \\
        /opt/venv/bin/python 04_nsight_systems/04_cuda_graph.py --nsys
    nsys stats -r cuda_api_sum outputs/graph.nsys-rep

运行：/opt/venv/bin/python 04_nsight_systems/04_cuda_graph.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common import ManySmallOps, ToyVLA, benchmark, compare, region  # noqa: E402


def make_graph(fn, static_inputs, warmup=3):
    """标准的 capture 流程：先在旁路 stream 上 warmup，再 capture。返回 (graph, static_output)。"""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):            # warmup：让 cuBLAS/cuDNN 完成初始化和选算法，不能在 capture 中做
            fn(*static_inputs)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):              # capture：只记录 kernel，不真正执行
        out = fn(*static_inputs)
    return g, out


def part_a():
    print("=== A. eager vs CUDA Graph ===")
    cases = []
    x = torch.randn(1024, device="cuda")
    cases.append(("ManySmallOps(200)", ManySmallOps(200).cuda(), (x,)))
    vla = ToyVLA().cuda().half().eval()
    cases.append(("ToyVLA fp16 batch=1", vla, (torch.randn(1, 3, 224, 224, device="cuda", dtype=torch.half),
                                               torch.randint(0, 32000, (1, 32), device="cuda"))))
    src = torch.empty(64 * 1024 * 1024, device="cuda")
    dst = torch.empty_like(src)

    def big_copy(_):
        for _ in range(4):
            dst.copy_(src)
        return dst
    cases.append(("256MB copy x4 (memory-bound)", big_copy, (src,)))

    for name, fn, inp in cases:
        with torch.no_grad():
            g, out = make_graph(fn, inp)
            with region(f"eager {name}"):
                r = [benchmark(lambda: fn(*inp), name="eager", warmup=10, iters=100)]
            with region(f"graph {name}"):
                r.append(benchmark(g.replay, name="CUDA Graph", warmup=10, iters=100))
        print(f"\n[{name}]")
        compare(r)


CAPTURE_FAIL = r"""
import sys, torch
sys.path.insert(0, %r)
from importlib import import_module
m = import_module("04_cuda_graph")
x = torch.randn(1024, device="cuda")
case = sys.argv[1]
fns = {
    "item": lambda t: t * 2 if t.sum().item() > 0 else t,   # .item() 同步：capture 期间不允许
    "nonzero": lambda t: t[t > 0],                            # 输出 shape 依赖数据 → 需要同步
    "cpu_tensor": lambda t: t + torch.ones(1024).cuda(),      # capture 期间做 H2D 拷贝（pageable）
}
try:
    m.make_graph(fns[case], (x,), warmup=1)
    print("OK capture 成功")
except Exception as e:
    print("FAIL", type(e).__name__, str(e).splitlines()[0][:120])
"""


def part_b():
    import subprocess
    print("\n=== B. capture 时的常见错误 ===")
    # 1) 动态 shape / 输入没有拷进静态缓冲区：graph 记录的是固定的指针和 shape，不会报错，而是"错得很安静"
    lin = torch.nn.Linear(64, 64).cuda()
    static_in = torch.randn(8, 64, device="cuda")
    with torch.no_grad():
        g, static_out = make_graph(lin, (static_in,))
        new = torch.randn(8, 64, device="cuda")
        static_in.copy_(new)                  # ✅ 正确做法：把新数据拷进 capture 时用的那块输入缓冲区
        g.replay()
        ok = torch.allclose(static_out, lin(new), atol=1e-5)
        other = torch.randn(8, 64, device="cuda")
        g.replay()                            # ❌ 错误做法：以为传了新输入，实际上 graph 读的还是 static_in
        wrong = torch.allclose(static_out, lin(other), atol=1e-5)
    print(f"  ✅ 新输入 copy_ 到静态缓冲区后 replay：结果正确 = {ok}")
    print(f"  ⚠️ 没有拷贝新输入就 replay：结果正确 = {wrong}（不报错，悄悄用旧输入计算）")

    # 2) 会导致 capture 失败的操作：失败后 CUDA context 会处于异常状态，所以每个用例在子进程里跑
    here = str(Path(__file__).resolve().parent)
    for case, desc in [("item", ".item() 同步"), ("nonzero", "布尔索引（数据相关 shape）"),
                       ("cpu_tensor", "capture 中创建 CPU 张量并拷到 GPU")]:
        out = subprocess.run([sys.executable, "-c", CAPTURE_FAIL % here, case], capture_output=True, text=True,
                             cwd=here)
        line = (out.stdout.strip().splitlines() or ["?"])[-1]
        print(f"  {'❌' if line.startswith('FAIL') else '✅'} {desc:<30s} {line}")


def nsys_mode():
    """配合 nsys 使用：各跑 20 次 eager 和 graph，用 NVTX 区分。"""
    vla = ToyVLA().cuda().half().eval()
    inp = (torch.randn(1, 3, 224, 224, device="cuda", dtype=torch.half), torch.randint(0, 32000, (1, 32), device="cuda"))
    with torch.no_grad():
        g, _ = make_graph(vla, inp)
        for _ in range(5):
            vla(*inp)
        torch.cuda.synchronize()
        with region("eager x20"):
            for _ in range(20):
                vla(*inp)
            torch.cuda.synchronize()
        with region("graph x20"):
            for _ in range(20):
                g.replay()
            torch.cuda.synchronize()


if __name__ == "__main__":
    if "--nsys" in sys.argv:
        nsys_mode()
    else:
        part_a()
        part_b()
