"""01-03 用系统级指标区分 5 种负载：空闲 / CPU-bound / Launch-bound / Memory-bound / Compute-bound。

依次运行 5 个阶段，同时用 JtopSampler 每 0.1s 采样一次；结束后输出：
  - 每阶段各项指标的平均值（判读表）
  - 时间线图 outputs/load_patterns.png（多个子图，x 轴对齐）
  - 原始数据 outputs/load_patterns.csv
  - Compute 阶段 GEMM 吞吐随时间的变化（功耗墙）

运行：/opt/venv/bin/python 01_system_monitor/03_load_patterns.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common import ManySmallOps, PipelineConfig, SlowPipeline, print_env  # noqa: E402
from common.monitor import JtopSampler  # noqa: E402

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)


def run_for(seconds, fn):
    """在 seconds 秒内反复调用 fn，返回 (调用次数, 实际耗时)。最后同步，保证 GPU 上的任务都已完成。"""
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        fn()
        n += 1
    torch.cuda.synchronize()
    return n, time.perf_counter() - t0


def main():
    print_env()
    # ---- 准备各阶段的负载 ----
    pipe = SlowPipeline(PipelineConfig.fast(py_preprocess=True))         # 只打开 Python 预处理这一个开关
    it = iter(range(10 ** 9))
    small = ManySmallOps(200).cuda()
    tiny = torch.randn(1024, device="cuda")
    x = torch.empty(64 * 1024 * 1024, device="cuda")
    y = torch.empty_like(x)
    a = torch.randn(8192, 8192, device="cuda", dtype=torch.float16)
    gemm_log = []  # (t, TFLOPS)

    def gemm_step():
        t = time.perf_counter()
        for _ in range(5):
            a @ a
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
        gemm_log.append((t, 5 * 2 * 8192 ** 3 / dt / 1e12))  # 绝对时间，事后再换算

    phases = [
        ("1 idle", 2.0, lambda: time.sleep(0.01)),
        ("2 CPU-bound", 4.0, lambda: pipe.step(next(it))),
        ("3 launch-bound", 4.0, lambda: small(tiny)),
        ("4 memory-bound", 4.0, lambda: y.copy_(x)),
        ("5 compute-bound", 6.0, gemm_step),
        ("6 cool-down", 2.0, lambda: time.sleep(0.01)),
    ]
    for _, _, fn in phases[1:5]:  # 预热（选算法、编译）
        fn()
    torch.cuda.synchronize()
    gemm_log.clear()

    stats = {}
    with JtopSampler(interval=0.1) as s:
        for name, dur, fn in phases:
            s.mark(name)
            stats[name] = run_for(dur, fn)

    gemm_log[:] = [(t - s.t0, v) for t, v in gemm_log]
    s.to_csv(OUT / "load_patterns.csv")
    fields = ["gpu_load", "cpu_max", "cpu_avg", "emc_load", "p_gpu_w", "p_cpu_soc_w", "p_tot_w", "temp_gpu"]
    print("\n=== 每阶段平均值 ===")
    print(s.summary(fields))
    print("\n=== 每阶段吞吐 ===")
    for name, (n, dt) in stats.items():
        print(f"{name:<18s} {n / dt:10.1f} iter/s   ({dt:.1f}s)")
    if gemm_log:
        first = [v for t, v in gemm_log[:3]]
        last = [v for t, v in gemm_log[-5:]]
        print(f"\nGEMM 吞吐：前 3 次 {sum(first) / len(first):.1f} TFLOPS → 最后 5 次 {sum(last) / len(last):.1f} TFLOPS")
    plot(s, gemm_log)
    print(f"\n图已保存: {OUT / 'load_patterns.png'}")


def plot(s, gemm_log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink2, grid, surface, c1 = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#2a78d6"
    shade = "#f0efec"
    t = [r["t"] for r in s.records]
    panels = [("gpu_load", "GPU load (%)  - time-based utilization", (0, 105)),
              ("cpu_max", "Busiest CPU core (%)", (0, 105)),
              ("p_gpu_w", "GPU rail power (W)", None),
              ("p_cpu_soc_w", "CPU/SoC rail power (W)", None)]
    fig, axes = plt.subplots(len(panels) + 1, 1, figsize=(11, 10), sharex=True, facecolor=surface)
    bounds = s.marks + [(t[-1], "")]
    for ax, (key, title, ylim) in zip(axes, panels):
        ax.plot(t, [r[key] for r in s.records], color=c1, lw=2)
        ax.set_title(title, loc="left", fontsize=10, color=ink)
        if ylim:
            ax.set_ylim(*ylim)
    ax = axes[-1]
    if gemm_log:
        ax.plot([g[0] for g in gemm_log], [g[1] for g in gemm_log], color=c1, lw=2, marker="o", ms=4)
    ax.set_title("FP16 GEMM 8192 throughput (TFLOPS) - compute phase only", loc="left", fontsize=10, color=ink)
    ax.set_xlabel("time (s)", color=ink)
    for ax in axes:
        ax.set_facecolor(surface)
        for i, ((t0, name), (t1, _)) in enumerate(zip(bounds[:-1], bounds[1:])):
            if i % 2 == 1:
                ax.axvspan(t0, t1, color=shade, zorder=0)
        ax.grid(True, axis="y", color=grid, lw=0.8)
        for sp in ax.spines.values():
            sp.set_color(grid)
        ax.tick_params(colors=ink2, labelsize=9)
    top = axes[0]
    top.set_title(panels[0][1], loc="left", fontsize=10, color=ink, pad=24)  # 给阶段标签留出空间
    for (t0, name), (t1, _) in zip(bounds[:-1], bounds[1:]):
        top.text((t0 + t1) / 2, 1.04, name, ha="center", va="bottom", fontsize=9, color=ink2,
                 transform=top.get_xaxis_transform())
    fig.suptitle("System-level signals for 5 workload types (jtop, 0.1 s sampling)", x=0.01, ha="left",
                 color=ink, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "load_patterns.png", dpi=120, facecolor=surface)


if __name__ == "__main__":
    main()
