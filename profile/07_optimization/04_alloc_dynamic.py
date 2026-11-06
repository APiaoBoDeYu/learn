"""07-04 显存分配与动态 shape

    /opt/venv/bin/python 07_optimization/04_alloc_dynamic.py

A. 逐步增长的分配（模拟动态 shape 的中间结果）：第 i 次申请 (i+1)*8 MB 并清零，释放上一次的
   default            PyTorch caching allocator（默认）
   expandable         PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   cudaMallocAsync    PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync（CUDA 驱动的显存池）
   prealloc           一次申请最大尺寸，之后只取 view
   allocator 配置必须在 CUDA 初始化之前设置，所以每种配置在单独的子进程中运行
B. 动态 shape + torch.compile：序列长度在 17~256 之间随机变化的 MLP + LayerNorm
   dynamic=None（默认：第二个新 shape 出现时自动转成动态 shape）/ dynamic=False（每个 shape 编译一次）/
   dynamic=False + 分桶 padding 到 {64,128,256}
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

OUT = Path(__file__).resolve().parent / "outputs"
lines = []


def p(s=""):
    print(s); lines.append(s)


# ---------------------------------------------------------------- A（子进程）
def worker_alloc(mode):
    import torch
    n_steps, mb = 60, 8
    torch.cuda.init()
    if mode == "prealloc":
        buf = torch.empty(n_steps * mb * 2 ** 20 // 4, device="cuda")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    prev = None
    for i in range(n_steps):
        n = (i + 1) * mb * 2 ** 20 // 4
        t = buf[:n] if mode == "prealloc" else torch.empty(n, device="cuda")
        t.zero_()
        prev = t  # noqa: F841  （释放上一次的张量）
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000
    st = torch.cuda.memory_stats()
    print(json.dumps({"ms": ms, "peak_alloc_MB": torch.cuda.max_memory_allocated() / 2 ** 20,
                      "peak_reserved_MB": torch.cuda.max_memory_reserved() / 2 ** 20,
                      "n_device_alloc": st.get("num_device_alloc", -1),
                      "logical_need_MB": 2 * n_steps * mb}))


def part_a():
    p("== A. 逐步增长的分配：60 步，第 i 步申请 (i+1)*8 MB（最大 480 MB）==")
    p(f"  {'allocator':<16s} {'总时间 ms':>9s} {'峰值 allocated':>14s} {'峰值 reserved':>13s} {'cudaMalloc 次数':>15s}")
    for mode, conf in (("default", ""), ("expandable", "expandable_segments:True"),
                       ("cudaMallocAsync", "backend:cudaMallocAsync"), ("prealloc", "")):
        env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": conf} if conf else {k: v for k, v in os.environ.items() if k != "PYTORCH_CUDA_ALLOC_CONF"}
        r = subprocess.run([sys.executable, __file__, "--worker-alloc", mode], capture_output=True, text=True, env=env)
        d = json.loads(r.stdout.strip().splitlines()[-1])
        p(f"  {mode:<16s} {d['ms']:9.1f} {d['peak_alloc_MB']:11.0f} MB {d['peak_reserved_MB']:10.0f} MB {d['n_device_alloc']:15d}")
    p("  （cudaMallocAsync 后端的分配走 CUDA 驱动的显存池（cudaMallocAsync），不计入 num_device_alloc，所以显示 0）")


# ---------------------------------------------------------------- B
def part_b():
    """在子进程中跑两次：第一次用全新的 inductor 缓存目录（冷启动），第二次复用同一目录（热缓存）。"""
    import tempfile
    cache = tempfile.mkdtemp(prefix="inductor_cache_")
    for tag in ("冷启动（全新的 TORCHINDUCTOR_CACHE_DIR）", "热缓存（同一目录再跑一次）"):
        r = subprocess.run([sys.executable, __file__, "--worker-dyn", tag], capture_output=True, text=True,
                           env={**os.environ, "TORCHINDUCTOR_CACHE_DIR": cache})
        for line in r.stdout.splitlines():
            p(line)


def worker_dyn(tag):
    import random
    import torch
    import torch._dynamo
    import torch.nn as nn
    from torch._dynamo.utils import counters

    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(512, 2048), nn.GELU(), nn.Linear(2048, 512), nn.LayerNorm(512)).cuda().half().eval()
    random.seed(0)
    lens = [random.randint(17, 256) for _ in range(60)]
    p(f"\n== B. 动态 shape：60 次调用，序列长度在 17~256 之间随机（{len(set(lens))} 种不同长度）；{tag} ==")
    p(f"  {'方案':<36s} {'编译次数':>8s} {'60 次总耗时 s':>13s} {'稳态 ms/次':>11s}")

    def bucket(L):
        return next(b for b in (64, 128, 256) if L <= b)

    for name, dyn, use_bucket in (("eager", "eager", False), ("compile dynamic=None（默认）", None, False),
                                  ("compile dynamic=False", False, False), ("compile dynamic=False + 分桶", False, True)):
        torch._dynamo.reset()
        counters.clear()
        f = m if dyn == "eager" else torch.compile(m, dynamic=dyn)
        xs = {L: torch.randn(1, L, 512, device="cuda", dtype=torch.half) for L in set(lens)}
        pads = {b: torch.zeros(1, b, 512, device="cuda", dtype=torch.half) for b in (64, 128, 256)}

        def call(L):
            if use_bucket:
                b = bucket(L)
                buf = pads[b]
                buf[:, :L].copy_(xs[L])
                return f(buf)[:, :L]
            return f(xs[L])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for L in lens:
                call(L)
        torch.cuda.synchronize()
        total = time.perf_counter() - t0
        n_comp = counters["stats"].get("unique_graphs", 0) if dyn != "eager" else 0
        with torch.no_grad():
            for L in lens[:5]:
                call(L)
            torch.cuda.synchronize(); t1 = time.perf_counter()
            for L in lens:
                call(L)
            torch.cuda.synchronize()
        steady = (time.perf_counter() - t1) / len(lens) * 1000
        p(f"  {name:<36s} {n_comp:8d} {total:13.2f} {steady:11.3f}")
    p(f"  （torch._dynamo.config.recompile_limit = {getattr(torch._dynamo.config, 'recompile_limit', getattr(torch._dynamo.config, 'cache_size_limit', '?'))}："
      "超过之后该函数会退回 eager）")


if __name__ == "__main__":
    if "--worker-alloc" in sys.argv:
        worker_alloc(sys.argv[-1]); sys.exit()
    if "--worker-dyn" in sys.argv:
        worker_dyn(sys.argv[-1]); sys.exit()
    part_a(); part_b()
    (OUT / "04_alloc_dynamic.log").write_text("\n".join(lines) + "\n")
