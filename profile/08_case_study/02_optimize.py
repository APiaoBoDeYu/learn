"""08-02 逐项修复，记录每一步的收益

    /opt/venv/bin/python 08_case_study/02_optimize.py

每个阶段（case.STAGES）测量：
    端到端延迟    200 步的 p50 / p99（wall，不开任何 profiler）
    各阶段拆分    在阶段边界同步后分别计时（只用于看"时间花在哪"，总和会略大于端到端延迟）
    功耗          jtop 连续运行 3 s 的平均总功耗 / GPU 功耗 → 每步能耗 = 功耗 × 每步时间
    MFU           5.59 GFLOP / p50 / FP16 峰值 155.6 TFLOPS
最后用 nsys 测最终阶段（S9）的 GPU busy，与 01_diagnose 的基线（34.2%）对比。
输出：outputs/02_optimize.log、outputs/02_optimize.json、outputs/stages.png
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import case  # noqa: E402

lines = []


def p(s=""):
    print(s); lines.append(s)


def worker_nsys(idx):
    """nsys 下运行第 idx 个阶段：warmup 后用 cudaProfilerStart/Stop 只采 20 步。"""
    from common import region
    pipe = case.STAGES[idx][1]()
    for i in range(20):
        pipe.step(i)
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    for i in range(20):
        with region(f"step_{i}"):
            pipe.step(i)
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()


def nsys_gpu_busy(idx):
    import importlib.util
    rep = case.OUT / f"stage{idx}_nsys"
    # --cuda-graph-trace=node：否则 CUDA Graph 内的 kernel 不会出现在 KERNEL 表中，GPU busy 会被严重低估（第 04 章常见坑 5）
    subprocess.run(["nsys", "profile", "-t", "cuda,nvtx", "--cuda-graph-trace=node", "--capture-range=cudaProfilerApi",
                    "--capture-range-end=stop",
                    "-o", str(rep), "--force-overwrite", "true", sys.executable, __file__, "--worker-nsys", str(idx)],
                   capture_output=True)
    subprocess.run(["nsys", "export", "-t", "sqlite", "-o", f"{rep}.sqlite", "--force-overwrite", "true", f"{rep}.nsys-rep"],
                   capture_output=True)
    spec = importlib.util.spec_from_file_location("an", case.ROOT / "04_nsight_systems" / "02_analyze_sqlite.py")
    an = importlib.util.module_from_spec(spec); spec.loader.exec_module(an)
    nvtx, gpu, _ = an.load(Path(f"{rep}.sqlite"))
    steps = [(s, e) for s, e, n in nvtx if n and n.startswith("step_")]
    lo, hi = steps[0][0], steps[-1][1]
    busy = an.union_len([(s, e) for s, e, _, _ in gpu], lo, hi)
    n_k = sum(1 for s, e, kind, _ in gpu if kind == "kernel" and lo <= s < hi)
    return busy / (hi - lo), (hi - lo) / len(steps) / 1e6, n_k / len(steps)


def plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    phases = [("preprocess", "#2a78d6"), ("infer", "#eb6834"), ("postprocess", "#1baf7a")]   # capture 只有约 0.002 ms，不画
    names = [r["name"].split(" ", 1)[0] + " " + r["name_en"] for r in res]
    fig, ax = plt.subplots(figsize=(10.5, 5.6), facecolor=surface)
    ax.set_facecolor(surface)
    y = np.arange(len(res))[::-1]
    left = np.zeros(len(res))
    for ph, c in phases:
        w = np.array([r["phases"][ph] for r in res])
        ax.barh(y, w, left=left, height=0.62, color=c, edgecolor=surface, linewidth=2, label=ph)
        left += w
    for yi, r in zip(y, res):
        ax.text(left[list(y).index(yi)] + 0.15, yi, f"{r['p50']:.2f} ms  ({res[0]['p50'] / r['p50']:.1f}x)", va="center",
                fontsize=8.5, color=ink)
    ax.set_yticks(y, names, fontsize=9, color=ink)
    ax.set_xlabel("time per step (ms)  - bars: per-phase median (synchronized); label: end-to-end p50", color=ink2)
    ax.set_xlim(0, max(left) * 1.28)
    ax.grid(True, axis="x", color=grid, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color(grid)
    ax.tick_params(colors=ink2)
    ax.legend(frameon=False, ncol=3, loc="lower right", fontsize=9, labelcolor=ink)
    ax.set_title("ToyVLA slow pipeline: step-by-step optimization - Jetson AGX Thor", color=ink, loc="left")
    fig.tight_layout()
    fig.savefig(case.OUT / "stages.png", dpi=130, facecolor=surface)


NAME_EN = ["baseline", "vectorized postprocess", "GPU normalize", "no .item() in loop", "fp16", "pinned + prealloc",
           "channels_last", "torch.compile reduce-overhead", "TensorRT fp16 + CUDA Graph", "GPU resize"]


def main():
    res = []
    p(f"{'阶段':<34s} {'p50 ms':>7s} {'p99 ms':>7s} {'累计':>6s} {'本步':>6s} | {'capture':>7s} {'preproc':>7s} {'infer':>7s} "
      f"{'postproc':>8s} | {'总功耗W':>7s} {'GPU W':>6s} {'mJ/step':>7s} {'MFU':>6s} {'GPU负载':>7s} {'CPU最忙核':>8s}")
    for k, (name, mk) in enumerate(case.STAGES):
        pipe = mk()
        ts = case.time_steps(pipe)
        ph = case.phase_times(pipe)
        pw = case.power_during(pipe, 3.0)
        p50, p99 = float(np.median(ts)), float(np.percentile(ts, 99))
        r = dict(name=name, name_en=NAME_EN[k], p50=p50, p99=p99, phases=ph, **pw)
        r["mJ"] = pw["p_tot_w"] * p50
        r["mfu"] = case.FLOP_PER_STEP / (p50 / 1000) / (case.PEAK_FP16_TFLOPS * 1e12)
        res.append(r)
        prev = res[-2]["p50"] if len(res) > 1 else p50
        p(f"{name:<34s} {p50:7.3f} {p99:7.3f} {res[0]['p50'] / p50:5.2f}x {prev / p50:5.2f}x | {ph['capture']:7.3f} "
          f"{ph['preprocess']:7.3f} {ph['infer']:7.3f} {ph['postprocess']:8.3f} | {pw['p_tot_w']:7.1f} {pw['p_gpu_w']:6.1f} "
          f"{r['mJ']:7.0f} {100 * r['mfu']:5.2f}% {pw['gpu_load']:6.0f}% {pw['cpu_max']:7.0f}%")
        del pipe
        torch.cuda.empty_cache()
    busy, ms, nk = nsys_gpu_busy(len(case.STAGES) - 1)
    p(f"\n最终阶段（{case.STAGES[-1][0]}）nsys：GPU busy {busy:.1%}，{ms:.2f} ms/step（nsys 下），每步 {nk:.0f} 个 kernel"
      f"（基线见 01_diagnose.log：GPU busy 34.2%，每步 517 个 kernel）")
    (case.OUT / "02_optimize.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    (case.OUT / "02_optimize.log").write_text("\n".join(lines) + "\n")
    plot(res)


if __name__ == "__main__":
    if "--worker-nsys" in sys.argv:
        worker_nsys(int(sys.argv[-1])); sys.exit()
    main()
