"""把本章 02~06 采集到的 kernel 画在 Thor 的实测 Roofline 上 → outputs/roofline_ncu.png

    python 06_plot_roofline.py        （先运行 06_tensor_core_roofline.py 生成 outputs/roofline_points.json）

横轴 AI = ncu 统计的 FLOP / 估算的 DRAM 字节；纵轴 = FLOP / ncu 测得的 kernel 时间。
屋顶取第 00 章实测值（新容器）：DRAM 240 GB/s，FP32 6.39 TFLOPS（CUDA core 理论 8.06），FP16 Tensor Core 155.6 TFLOPS。
配色：dataviz 参考调色板前 5 个类别色，已用 validate_palette.js 校验（对比度 WARN → 每个点都直接标注，并用 marker 形状做第二编码）。
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ncu_tools as T  # noqa: E402

BW = T.PEAK_BW_GBS
ROOFS = [("FP16 Tensor Core 155.6 TF (measured)", 155.6, "-", 1.15),
         ("FP32 8.06 TF (theoretical)", 8.06, ":", 1.12),
         ("FP32 6.39 TF (measured)", 6.39, "--", 0.62)]
SERIES = {"02": ("02 memory-bound", "#2a78d6", "o"), "03": ("03 register-bound", "#eb6834", "s"),
          "04": ("04 control-flow", "#1baf7a", "^"), "05": ("05 transpose / bank", "#eda100", "D"),
          "06": ("06 GEMM", "#e87ba4", "X")}
# 标注：(dx, dy, ha)，单位 offset points；None = 与旁边的点合并标注
LABELS = {
    "eager_3ops": ("eager_3ops (each of 3)", -8, -3, "right"), "torch_mul": None, "vec4": None,
    "coalesced": ("coalesced / vec4 / torch_mul", -8, 3, "right"), "fused": ("fused", -8, 4, "right"),
    "strided": ("strided", 8, -4, "left"),
    "A_per_thread": ("A_per_thread", 8, 0, "left"), "A_row_per_thread": ("A_row_per_thread", 8, -10, "left"),
    "A_per_thread_r64": ("A_per_thread_r64 (spill)", 9, -8, "left"), "A_torch_bmm": ("A_torch_bmm", 8, -4, "left"),
    "B_per_thread": ("B_per_thread / B_split4", -9, -4, "right"), "B_split4": None,
    "B_prefetch": None, "B_split4_prefetch": ("B_prefetch / B_split4_prefetch", -9, -1, "right"),
    "A_divergent": ("A_divergent", 9, -4, "left"), "A_warp_uniform": ("A_warp_uniform", 9, 3, "left"),
    "B_random": ("B_random", 0, -15, "center"), "B_sorted": ("B_sorted / B_uniform_mean", -2, 17, "center"),
    "B_uniform_mean": None,
    "fp32": ("GEMM fp32", 0, -13, "center"), "tf32": ("GEMM tf32", -9, 0, "right"), "fp16": ("GEMM fp16 / bf16", -9, 2, "right"),
    "bf16": None, "fp16_K4095": ("GEMM fp16, K=4095", 9, -4, "left"),
}


def main():
    pts = json.loads((T.OUT / "roofline_points.json").read_text())
    ink, ink2, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    fig, ax = plt.subplots(figsize=(11, 7), facecolor=surface)
    ax.set_facecolor(surface)
    xs = [10 ** (i / 20) for i in range(-40, 81)]
    for name, tf, ls, k in ROOFS:
        ax.plot(xs, [min(tf * 1e3, BW * x) for x in xs], color=ink2, lw=1.5, ls=ls)
        ax.text(xs[-1], tf * 1e3 * k, name, color=ink2, ha="right", va="bottom" if k > 1 else "top", fontsize=9)
    ax.text(0.03, BW * 0.03 * 0.35, f"DRAM roof (measured) {BW:.0f} GB/s", color=ink2, fontsize=9)
    for p in pts:
        label, c, mk = SERIES[p["chapter"]]
        x, y = p["ai"], p["flops"] / 1e9
        ax.scatter(x, y, s=60, color=c, marker=mk, edgecolors=surface, linewidths=2, zorder=3, label=label)
        lab = LABELS.get(p["variant"], (p["variant"], 8, -4, "left"))
        if lab:
            text, dx, dy, ha = lab
            ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, dy), ha=ha, va="center",
                        fontsize=7.5, color=ink)
    h, lab = ax.get_legend_handles_labels()
    uniq = dict(zip(lab, h))
    ax.legend(uniq.values(), uniq.keys(), frameon=False, loc="upper left", labelcolor=ink, fontsize=9)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(xs[0], xs[-1] * 1.2); ax.set_ylim(1, 5e5)
    ax.set_xlabel("Arithmetic Intensity (FLOP / DRAM Byte, from ncu counters)", color=ink)
    ax.set_ylabel("Achieved performance (GFLOP/s, ncu kernel duration)", color=ink)
    ax.set_title("Chapter 05 kernels on the measured roofline - Jetson AGX Thor", color=ink, loc="left")
    ax.grid(True, which="major", color=grid, lw=0.8)
    for s in ax.spines.values():
        s.set_color(grid)
    ax.tick_params(colors=ink2)
    fig.tight_layout()
    out = T.OUT / "roofline_ncu.png"
    fig.savefig(out, dpi=130, facecolor=surface)
    print("saved", out)


if __name__ == "__main__":
    main()
