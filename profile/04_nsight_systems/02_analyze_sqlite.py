"""04-02 直接用 SQL 分析 nsys 报告（nsys export -t sqlite），并画出 CPU/GPU 时间线。

对每个阶段（NVTX range）统计：
  - CPU 时间：range 的墙钟时长
  - GPU busy：该时间段内所有 GPU 操作（kernel/memcpy/memset）区间的并集
  - kernel 数、短 kernel（< 5us）占比 → Launch-bound 的证据
  - launch 延迟：cudaLaunchKernel 调用开始 → kernel 在 GPU 上开始执行（通过 correlationId 关联）
最后画出 slow/fast 各 2 个 step 的时间线：outputs/timeline.png

运行：/opt/venv/bin/python 04_nsight_systems/02_analyze_sqlite.py   （需要先运行 01_nsys_basics.sh）
"""
import sqlite3
import sys
from pathlib import Path

OUT = Path(__file__).parent / "outputs"
STAGES = ["preprocess", "infer", "postprocess"]


def load(db: Path):
    c = sqlite3.connect(db)
    nvtx = c.execute("""SELECT start, end, COALESCE(text, (SELECT value FROM StringIds WHERE id = textId))
                        FROM NVTX_EVENTS WHERE eventType = 59 AND end IS NOT NULL ORDER BY start""").fetchall()
    gpu = []
    for tbl, kind in [("CUPTI_ACTIVITY_KIND_KERNEL", "kernel"), ("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy"),
                      ("CUPTI_ACTIVITY_KIND_MEMSET", "memset")]:
        try:
            gpu += [(s, e, kind, cid) for s, e, cid in c.execute(f"SELECT start, end, correlationId FROM {tbl}")]
        except sqlite3.OperationalError:
            pass
    gpu.sort()
    launch = dict(c.execute("""SELECT r.correlationId, r.start FROM CUPTI_ACTIVITY_KIND_RUNTIME r
                               JOIN StringIds s ON r.nameId = s.id WHERE s.value LIKE 'cudaLaunchKernel%'
                               OR s.value LIKE 'cuLaunchKernel%'"""))
    return nvtx, gpu, launch


def union_len(intervals, lo, hi) -> float:
    """区间并集长度（裁剪到 [lo, hi]）。"""
    tot, cur_s, cur_e = 0, None, None
    for s, e in sorted((max(s, lo), min(e, hi)) for s, e in intervals if e > lo and s < hi):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                tot += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        tot += cur_e - cur_s
    return tot


def analyze(tag: str):
    nvtx, gpu, launch = load(OUT / f"pipeline_{tag}.sqlite")
    steps = [(s, e) for s, e, n in nvtx if n and n.startswith("step_")]
    print(f"\n=== {tag}: {len(steps)} steps ===")
    lo, hi = steps[0][0], steps[-1][1]
    wall = hi - lo
    busy = union_len([(s, e) for s, e, _, _ in gpu], lo, hi)
    print(f"step 平均 {wall / len(steps) / 1e6:.3f} ms，GPU busy {busy / wall:.1%}（kernel/memcpy/memset 区间并集 ÷ 墙钟）")

    print(f"{'stage':<12s}{'CPU ms':>8s}{'GPU busy ms':>12s}{'kernels':>9s}{'<5us':>7s}{'launch→start':>14s}")
    for st in STAGES:
        rs = [(s, e) for s, e, n in nvtx if n == st and lo <= s < hi]
        cpu = sum(e - s for s, e in rs)
        ks = [(s, e, cid) for s, e, k, cid in gpu if k == "kernel" and cid in launch
              and any(a <= launch[cid] < b for a, b in rs)]          # 由该阶段发起的 kernel
        gbusy = union_len([(s, e) for s, e, _ in ks], lo, hi)
        short = sum(1 for s, e, _ in ks if e - s < 5000)
        lat = sorted(s - launch[cid] for s, e, cid in ks)
        med = lat[len(lat) // 2] / 1000 if lat else 0
        n = len(rs)
        print(f"{st:<12s}{cpu / n / 1e6:8.3f}{gbusy / n / 1e6:12.3f}{len(ks) / n:9.0f}"
              f"{(short / len(ks) if ks else 0):7.0%}{med:12.1f}us")
    return nvtx, gpu, steps


def plot(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    ink, ink2, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    col = {"preprocess": "#2a78d6", "infer": "#eb6834", "postprocess": "#1baf7a"}
    gcol = {"kernel": "#52514e", "memcpy": "#e34948", "memset": "#eda100"}
    fig, axes = plt.subplots(len(data), 1, figsize=(12, 2.4 * len(data) + 0.6), facecolor=surface)
    for ax, (tag, (nvtx, gpu, steps)) in zip(axes, data.items()):
        lo, hi = steps[5][0], steps[6][1]                 # 只画第 6、7 两个 step
        for s, e, n in nvtx:
            if n in col and s < hi and e > lo:
                ax.broken_barh([((s - lo) / 1e6, (e - s) / 1e6)], (1.1, 0.8), color=col[n], lw=0)
        for kind in gcol:
            bars = [((s - lo) / 1e6, max((e - s) / 1e6, 0.004)) for s, e, k, _ in gpu
                    if k == kind and s < hi and e > lo]
            ax.broken_barh(bars, (0.1, 0.8), color=gcol[kind], lw=0)
        busy = union_len([(s, e) for s, e, _, _ in gpu], lo, hi) / (hi - lo)
        ax.set_yticks([0.5, 1.5], ["GPU", "CPU (NVTX)"])
        ax.set_ylim(0, 2)
        ax.set_xlim(0, (hi - lo) / 1e6)
        ax.set_title(f"{tag} pipeline: 2 steps = {(hi - lo) / 1e6:.1f} ms, GPU busy {busy:.0%}",
                     loc="left", fontsize=10, color=ink)
        ax.set_facecolor(surface)
        ax.grid(True, axis="x", color=grid, lw=0.8)
        for sp in ax.spines.values():
            sp.set_color(grid)
        ax.tick_params(colors=ink2, labelsize=9)
    axes[-1].set_xlabel("time (ms)", color=ink)
    handles = [Patch(color=c, label=f"CPU: {n}") for n, c in col.items()] + \
              [Patch(color=c, label=f"GPU: {k}") for k, c in gcol.items()]
    fig.legend(handles=handles, loc="upper right", ncol=6, frameon=False, fontsize=8, labelcolor=ink)
    fig.suptitle("nsys timeline (from sqlite export)", x=0.01, ha="left", fontsize=11, color=ink)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "timeline.png", dpi=120, facecolor=surface)
    print(f"\n图已保存: {OUT / 'timeline.png'}")


def main():
    tags = sys.argv[1:] or ["slow", "fast"]
    data = {t: analyze(t) for t in tags}
    plot(data)


if __name__ == "__main__":
    main()
