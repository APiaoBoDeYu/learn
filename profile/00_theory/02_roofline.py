"""00-02 在 Thor 上实测 Roofline：峰值带宽、峰值算力，以及典型算子在 Roofline 上的位置。

运行：/opt/venv/bin/python 00_theory/02_roofline.py
产物：00_theory/outputs/roofline.png、roofline.csv
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from common import benchmark, print_env  # noqa: E402

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)

# 理论峰值：FP32 CUDA core = SM 数 × 每 SM 128 个 FP32 单元 × 2(FMA) × 频率
props = torch.cuda.get_device_properties(0)
GPU_GHZ = 1.575  # 由 jtop 读到（jetson_clocks 锁频）
THEO_FP32_TFLOPS = props.multi_processor_count * 128 * 2 * GPU_GHZ / 1e3


_FLUSH = None


def flush_l2():
    """写一块比 L2（Thor 为 32MB）大得多的缓冲区，把 L2 里的旧数据挤掉。"""
    global _FLUSH
    if _FLUSH is None:
        _FLUSH = torch.empty(256 * 1024 * 1024 // 4, device="cuda")
    _FLUSH.zero_()


def gpu_time(fn, iters=30, cold=True) -> float:
    """CUDA Event 计时取中位数。cold=True 时每次计时前刷新 L2，测到的是 DRAM 而不是 L2。"""
    if not cold:
        return benchmark(fn, mode="event", warmup=5, iters=iters).p50
    for _ in range(3):
        fn()
    ts = []
    for _ in range(iters):
        flush_l2()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e))
    return sorted(ts)[len(ts) // 2]


def measure_bandwidth() -> float:
    n = 64 * 1024 * 1024  # 64M float = 256 MB
    x = torch.empty(n, device="cuda")
    y = torch.empty(n, device="cuda")
    ms = gpu_time(lambda: y.copy_(x), cold=False)  # 256MB 远大于 L2，不需要刷
    return 2 * n * 4 / ms / 1e6  # GB/s（读 + 写）


def measure_matmul(dtype, sizes=(2048, 4096, 8192), tf32=False) -> float:
    """屋顶 = 各尺寸 GEMM 单次迭代的最好成绩（突发峰值）。

    为什么不用中位数：Thor 在持续的 Tensor Core 满载下会撞到功耗墙（GPU rail 约 96W），
    吞吐从约 155 降到约 100 TFLOPS，中位数随负载时长波动很大；单次最好成绩则很稳定。
    """
    torch.backends.cuda.matmul.allow_tf32 = tf32
    best = 0.0
    for n in sizes:
        a = torch.randn(n, n, device="cuda", dtype=dtype)
        b = torch.randn(n, n, device="cuda", dtype=dtype)
        st = benchmark(lambda: a @ b, mode="event", warmup=10, iters=30)
        best = max(best, 2 * n ** 3 / st.min / 1e9)
    torch.backends.cuda.matmul.allow_tf32 = False
    return best  # TFLOPS


def measure_sustained_fp16(seconds=5.0) -> float:
    """持续满载 seconds 秒的平均吞吐（会触发功耗墙）。"""
    import time
    a = torch.randn(8192, 8192, device="cuda", dtype=torch.float16)
    b = torch.randn_like(a)
    torch.cuda.synchronize()
    t0, n = time.perf_counter(), 0
    while time.perf_counter() - t0 < seconds:
        for _ in range(10):
            a @ b
        torch.cuda.synchronize()
        n += 10
    return 2 * 8192 ** 3 * n / (time.perf_counter() - t0) / 1e12


def op_points():
    """返回 [(名称, 类别, FLOPs, Bytes, 实测 ms)]。FLOPs/Bytes 按理论最小值计算（每个元素只读写一次）。"""
    pts = []
    n = 32 * 1024 * 1024
    a = torch.randn(n, device="cuda")
    b = torch.randn(n, device="cuda")
    pts.append(("add fp32", "elementwise / reduction", n, 12 * n, gpu_time(lambda: a + b)))
    pts.append(("relu fp32", "elementwise / reduction", n, 8 * n, gpu_time(lambda: torch.relu(a))))
    c = torch.randn(n, device="cuda")
    pts.append(("a*b+c fp32", "elementwise / reduction", 2 * n, 16 * n, gpu_time(lambda: torch.addcmul(c, a, b))))

    h = torch.randn(4096, 8192, device="cuda", dtype=torch.float16)
    m = h.numel()
    pts.append(("softmax fp16", "elementwise / reduction", 5 * m, 4 * m, gpu_time(lambda: torch.softmax(h, -1))))
    w = torch.ones(8192, device="cuda", dtype=torch.float16)
    pts.append(("layernorm fp16", "elementwise / reduction", 8 * m, 4 * m,
                gpu_time(lambda: F.layer_norm(h, (8192,), w, w))))

    # GEMV：batch=1 的 Linear，相当于 LLM decode
    W = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    v = torch.randn(1, 4096, device="cuda", dtype=torch.float16)
    gemv_bytes = 2 * (4096 * 4096 + 2 * 4096)
    pts.append(("GEMV (cold L2)", "GEMM fp16", 2 * 4096 * 4096, gemv_bytes, gpu_time(lambda: v @ W)))
    # 反例：不刷 L2，32MB 的权重常驻 L2，"带宽"远超 DRAM 峰值
    pts.append(("GEMV (hot L2!)", "GEMM fp16", 2 * 4096 * 4096, gemv_bytes, gpu_time(lambda: v @ W, cold=False)))
    for s in [256, 512, 1024, 2048, 4096, 8192]:
        A = torch.randn(s, s, device="cuda", dtype=torch.float16)
        B = torch.randn(s, s, device="cuda", dtype=torch.float16)
        pts.append((f"GEMM {s}", "GEMM fp16", 2 * s ** 3, 2 * 3 * s * s, gpu_time(lambda: A @ B, iters=10)))
    for s in [512, 2048, 8192]:
        A = torch.randn(s, s, device="cuda")
        B = torch.randn(s, s, device="cuda")
        pts.append((f"GEMM {s} fp32", "GEMM fp32", 2 * s ** 3, 4 * 3 * s * s, gpu_time(lambda: A @ B, iters=10)))
    return pts


def plot(bw, roofs, pts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 容器内没有中文字体，图中统一使用英文标签

    ink, ink2, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    series = {"elementwise / reduction": ("#2a78d6", "o"), "GEMM fp16": ("#eb6834", "s"), "GEMM fp32": ("#1baf7a", "^")}

    fig, ax = plt.subplots(figsize=(10, 6.5), facecolor=surface)
    ax.set_facecolor(surface)
    xs = [10 ** (i / 20) for i in range(-30, 81)]  # AI 从 0.03 到 1e4
    for name, tflops, style in roofs:
        ys = [min(tflops * 1e3, bw * x) for x in xs]  # GFLOPS
        ax.plot(xs, ys, color=ink2, lw=1.5, ls=style)
        ax.text(xs[-1] * 1.4, tflops * 1e3 * 1.12, name, color=ink2, ha="right", va="bottom", fontsize=9)
    ax.set_xlim(xs[0], xs[-1] * 1.5)
    ax.set_ylim(3, 4e5)
    ax.text(10, bw * 10 / 8, f"DRAM roof (measured) {bw:.0f} GB/s", color=ink2, fontsize=9)
    offsets = {"a*b+c fp32": (8, -4), "add fp32": (8, -6), "GEMM 4096": (-10, -18),
               "GEMM 256": (8, 2), "GEMM 512 fp32": (8, -12), "GEMV (cold L2)": (8, -12),
               "GEMM 8192 fp32": (8, -12), "GEMM 512": (-58, 2)}
    rename = {"relu fp32": None, "a*b+c fp32": "a*b+c / relu fp32"}  # 两点几乎重合，合并标注
    for name, cat, fl, by, ms in pts:
        c, mk = series[cat]
        ai, perf = fl / by, fl / ms / 1e6  # FLOP/B, GFLOPS
        ax.scatter(ai, perf, s=50, color=c, marker=mk, edgecolors=surface, linewidths=2, zorder=3,
                   label=cat)
        label = rename.get(name, name)
        if label:
            ax.annotate(label, (ai, perf), textcoords="offset points", xytext=offsets.get(name, (8, -10)),
                        fontsize=8, color=ink)

    h, lab = ax.get_legend_handles_labels()
    uniq = dict(zip(lab, h))
    ax.legend(uniq.values(), uniq.keys(), frameon=False, loc="upper left", labelcolor=ink)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity (FLOP / Byte)", color=ink)
    ax.set_ylabel("Achieved performance (GFLOPS)", color=ink)
    ax.set_title("Measured roofline - Jetson AGX Thor (MAXN, GPU 1575 MHz)", color=ink, loc="left")
    ax.grid(True, which="major", color=grid, lw=0.8)
    for s in ax.spines.values():
        s.set_color(grid)
    ax.tick_params(colors=ink2)
    fig.tight_layout()
    fig.savefig(OUT / "roofline.png", dpi=130, facecolor=surface)


def main():
    print_env()
    bw = measure_bandwidth()
    fp32 = measure_matmul(torch.float32)
    tf32 = measure_matmul(torch.float32, tf32=True)
    fp16 = measure_matmul(torch.float16)
    bf16 = measure_matmul(torch.bfloat16)
    sus = measure_sustained_fp16()
    print(f"\n峰值带宽 (copy 256MB)        : {bw:8.1f} GB/s")
    print(f"FP32 理论峰值 (CUDA core)     : {THEO_FP32_TFLOPS:8.2f} TFLOPS  "
          f"= {props.multi_processor_count} SM x 128 x 2 x {GPU_GHZ} GHz")
    print(f"FP32 matmul 实测(最优尺寸)    : {fp32:8.2f} TFLOPS  ({fp32 / THEO_FP32_TFLOPS:.0%} of 理论)")
    print(f"TF32 matmul 实测(最优尺寸)    : {tf32:8.2f} TFLOPS")
    print(f"FP16 matmul 实测(最优尺寸)    : {fp16:8.2f} TFLOPS")
    print(f"BF16 matmul 实测(最优尺寸)    : {bf16:8.2f} TFLOPS")
    print(f"FP16 持续满载 5s (8192)       : {sus:8.2f} TFLOPS  <- 功耗墙，见 README")
    print(f"拐点（FP16 屋顶 / 带宽）       : {fp16 * 1e3 / bw:8.1f} FLOP/B  <- AI 低于它就是 memory-bound")
    print(f"拐点（FP32 屋顶 / 带宽）       : {fp32 * 1e3 / bw:8.1f} FLOP/B")

    pts = op_points()
    print(f"\n{'op':<20s}{'AI(FLOP/B)':>12s}{'ms':>10s}{'GFLOPS':>10s}{'GB/s':>9s}  判定")
    rows = []
    for name, cat, fl, by, ms in pts:
        ai, gflops, gbs = fl / by, fl / ms / 1e6, by / ms / 1e6
        roof = fp16 if "fp16" in name or cat == "GEMM fp16" else fp32
        bound = "memory" if ai < roof * 1e3 / bw else "compute"
        eff = gbs / bw if bound == "memory" else gflops / (roof * 1e3)
        print(f"{name:<20s}{ai:12.2f}{ms:10.3f}{gflops:10.1f}{gbs:9.1f}  {bound}-bound, 达到屋顶 {eff:.0%}")
        rows.append(dict(op=name, category=cat, flops=fl, bytes=by, ms=round(ms, 4), ai=round(ai, 3),
                         gflops=round(gflops, 1), gbps=round(gbs, 1), bound=bound))
    with open(OUT / "roofline.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    roofs = [(f"FP16 tensor core peak (measured) {fp16:.1f} TFLOPS", fp16, "-"),
             (f"FP32 CUDA core peak (measured) {fp32:.1f} TFLOPS", fp32, "--")]
    plot(bw, roofs, pts)
    print(f"\n图已保存: {OUT / 'roofline.png'}")


if __name__ == "__main__":
    main()
