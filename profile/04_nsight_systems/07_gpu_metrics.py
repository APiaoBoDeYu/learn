"""04-07 nsys GPU metrics（需要 GPU 计数器权限：privileged 容器或 CAP_SYS_ADMIN）

    /opt/venv/bin/python 04_nsight_systems/07_gpu_metrics.py

nsys profile --gpu-metrics-devices=all --gpu-metrics-frequency=20000（每 50 us 采样一次硬件计数器）
Thor（GB10B）默认使用 gb10b 指标集：GR Active、SMs Active、SM Issue、Tensor Active、Compute Warps in Flight、
Copy Engine Active 等。注意：**没有 DRAM 带宽**（与 ncu 没有 dram__* 指标的原因相同，见第 05 章 5.3）。

A. 慢版 / 快版流水线：按 NVTX 阶段统计"空间维度"利用率（呼应第 00 章 0.3），画时间线 outputs/gpu_metrics.png
B. 4.5 节"H2D 拷贝和 kernel 严格交替"：拷贝期间 Copy Engine 和 SM 各在做什么；D2D 拷贝能否与 kernel 重叠
C. 拷贝由谁执行（Copy Engine 还是 SM）与拷贝大小的关系；走 Copy Engine 的小拷贝能否与 GEMM 重叠、值不值得
D. 观察者效应：nsys 下慢版的 pageable H2D 拷贝偶尔要约 4 ms，不开 nsys 时不存在
"""
import sqlite3
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
PY = sys.executable
GM = ["--gpu-metrics-devices=all", "--gpu-metrics-frequency=20000"]
SHOW = ["GR Active [Throughput %]", "SMs Active [Throughput %]", "SM Issue [Throughput %]", "Tensor Active [Throughput %]",
        "Compute Warps in Flight [Throughput %]", "Async Copy Engine Active 0 [Throughput %]",
        "Async Copy Engine Active 1 [Throughput %]"]
SHORT = {"GR Active [Throughput %]": "GR", "SMs Active [Throughput %]": "SMs", "SM Issue [Throughput %]": "Issue",
         "Tensor Active [Throughput %]": "Tensor", "Compute Warps in Flight [Throughput %]": "Warps",
         "Async Copy Engine Active 0 [Throughput %]": "CE0", "Async Copy Engine Active 1 [Throughput %]": "CE1"}


# ---------------------------------------------------------------- worker（B 的负载）
def worker_overlap():
    import torch
    n = 8
    host = [torch.randn(8 * 1024 * 1024).pin_memory() for _ in range(n)]          # 32 MB/块
    dev = [torch.empty(8 * 1024 * 1024, device="cuda") for _ in range(n)]
    src = [torch.randn(8 * 1024 * 1024, device="cuda") for _ in range(n)]
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.half)
    s_copy, s_comp = torch.cuda.Stream(), torch.cuda.Stream()

    def run(kind):
        with torch.cuda.stream(s_copy):
            for i in range(n):
                if kind == "h2d":
                    dev[i].copy_(host[i], non_blocking=True)    # cudaMemcpyAsync HtoD
                else:
                    dev[i].copy_(src[i])                        # cudaMemcpyAsync DtoD
        with torch.cuda.stream(s_comp):
            for i in range(n):
                a @ a
        torch.cuda.synchronize()

    for k in ("h2d", "d2d"):
        run(k)
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    for k in ("h2d", "d2d"):
        torch.cuda.nvtx.range_push(f"overlap_{k}")
        for _ in range(3):
            run(k)
        torch.cuda.nvtx.range_pop()
    torch.cuda.profiler.stop()


SIZES = [64 << 10, 256 << 10, 512 << 10, 1 << 20, 4 << 20, 16 << 20, 32 << 20]


def worker_sizes():
    import torch
    H = {s: torch.randn(s // 4).pin_memory() for s in SIZES}
    D = {s: torch.empty(s // 4, device="cuda") for s in SIZES}
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.half)
    sc, sk = torch.cuda.Stream(), torch.cuda.Stream()
    for s in SIZES:
        D[s].copy_(H[s], non_blocking=True)
    a @ a
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    for s in SIZES:                                    # 单独拷贝：每种大小合计 >= 8 MB
        torch.cuda.nvtx.range_push(f"alone_{s >> 10}KB")
        for _ in range(max(4, (8 << 20) // s)):
            D[s].copy_(H[s], non_blocking=True)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
    for s in (64 << 10, 256 << 10, 16 << 20):           # 与 8 次 GEMM4096 并发：拷贝总量都是 128 MB
        torch.cuda.nvtx.range_push(f"withgemm_{s >> 10}KB")
        with torch.cuda.stream(sk):
            for _ in range(8):
                a @ a
        with torch.cuda.stream(sc):
            for _ in range((128 << 20) // s):
                D[s].copy_(H[s], non_blocking=True)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
    torch.cuda.profiler.stop()


# ---------------------------------------------------------------- 采集
def nsys(name, args):
    rep = OUT / f"{name}.nsys-rep"
    cmd = ["nsys", "profile", "-t", "cuda,nvtx", *GM, "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
           "-o", str(rep.with_suffix("")), "--force-overwrite", "true", PY, *args]
    r = subprocess.run(cmd, capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if "steps," in line:
            print("   ", line)
    subprocess.run(["nsys", "export", "-t", "sqlite", "-o", str(rep.with_suffix(".sqlite")), "--force-overwrite", "true",
                    str(rep)], capture_output=True)
    return rep.with_suffix(".sqlite")


def load(db):
    c = sqlite3.connect(db)
    names = {mid: n for _, _, _, mid, n in c.execute("SELECT * FROM TARGET_INFO_GPU_METRICS")}
    ids = {n: i for i, n in names.items()}
    samples = {}
    for ts, mid, v in c.execute("SELECT timestamp, metricId, value FROM GPU_METRICS ORDER BY timestamp"):
        samples.setdefault(ts, {})[names[mid]] = v
    nvtx = c.execute("""SELECT start, end, COALESCE(text, (SELECT value FROM StringIds WHERE id = textId))
                        FROM NVTX_EVENTS WHERE eventType = 59 AND end IS NOT NULL ORDER BY start""").fetchall()
    gpu = {}
    for tbl, kind in [("CUPTI_ACTIVITY_KIND_KERNEL", "kernel"), ("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy")]:
        try:
            gpu[kind] = c.execute(f"SELECT start, end FROM {tbl} ORDER BY start").fetchall()
        except sqlite3.OperationalError:
            gpu[kind] = []
    return sorted(samples.items()), nvtx, gpu, ids


def mean(rows, key):
    xs = [r[key] for r in rows if key in r]
    return sum(xs) / len(xs) if xs else float("nan")


# ---------------------------------------------------------------- A
def part_a():
    print("=== A. 流水线：按阶段统计 GPU metrics（20 kHz 采样）===")
    res = {}
    for tag, args in (("slow", []), ("fast", ["--fast"])):
        db = nsys(f"gm_{tag}", [str(HERE / "workload_nvtx.py"), "--steps", "20", *args])
        samples, nvtx, gpu, _ = load(db)
        steps = [(s, e) for s, e, n in nvtx if n and n.startswith("step_")]
        lo, hi = steps[0][0], steps[-1][1]
        phases = [(s, e, n) for s, e, n in nvtx if n in ("capture", "preprocess", "infer", "postprocess") and lo <= s < hi]
        inside = [v for t, v in samples if lo <= t < hi]
        print(f"\n[{tag}] {len(inside)} 个样本（{len(steps)} 步）")
        print(f"  {'阶段':<12s} {'样本':>5s} " + " ".join(f"{SHORT[k]:>7s}" for k in SHOW))
        by_phase = {}
        j = 0
        for t, v in samples:
            if not lo <= t < hi:
                continue
            while j < len(phases) and phases[j][1] <= t:
                j += 1
            if j < len(phases) and phases[j][0] <= t:
                by_phase.setdefault(phases[j][2], []).append(v)
        for ph in ("capture", "preprocess", "infer", "postprocess"):
            rows = by_phase.get(ph, [])
            print(f"  {ph:<12s} {len(rows):5d} " + " ".join(f"{mean(rows, k):7.1f}" for k in SHOW))
        print(f"  {'全部':<12s} {len(inside):5d} " + " ".join(f"{mean(inside, k):7.1f}" for k in SHOW))
        gr, sms = mean(inside, SHOW[0]), mean(inside, SHOW[1])
        print(f"  → GR Active（时间维度）{gr:.1f}%；SMs Active {sms:.1f}%（Gross，按全部时间平均）；"
              f"GPU 忙的时候平均有 {100 * sms / gr:.0f}% 的 SM 在工作（Net = SMs/GR）；"
              f"SM Issue {mean(inside, SHOW[2]):.1f}%（真正发射指令的比例）")
        res[tag] = (samples, phases, steps)
    return res


def plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    series = [("SMs Active [Throughput %]", "SMs Active", "#2a78d6"), ("SM Issue [Throughput %]", "SM Issue", "#eb6834"),
              ("Tensor Active [Throughput %]", "Tensor Active", "#1baf7a")]
    shade = {"preprocess": "#ecebe7", "postprocess": "#f6f5f2"}
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.2), facecolor=surface, sharey=True)
    for ax, tag in zip(axes, ("slow", "fast")):
        samples, phases, steps = res[tag]
        lo, hi = steps[5][0], steps[6][1]                        # 第 6、7 步
        ax.set_facecolor(surface)
        for s, e, n in phases:
            if not lo <= s < hi:
                continue
            if n in shade:
                ax.axvspan((s - lo) / 1e6, (e - lo) / 1e6, color=shade[n], lw=0)
            if n in ("preprocess", "infer", "postprocess") and (e - s) > 0.06 * (hi - lo):   # 太窄的阶段不标注
                ax.text(((s + e) / 2 - lo) / 1e6, 103, n, ha="center", fontsize=7.5, color=ink2)
        pts = [(t, v) for t, v in samples if lo <= t < hi]
        for key, label, c in series:
            ax.plot([(t - lo) / 1e6 for t, _ in pts], [v.get(key, 0) for _, v in pts], color=c, lw=1.2, label=label)
        ax.set_title(f"{tag} pipeline (2 steps)", color=ink, loc="left", fontsize=10)
        ax.set_ylim(0, 112)
        ax.set_ylabel("% of peak", color=ink2)
        ax.grid(True, axis="y", color=grid, lw=0.8)
        for sp in ax.spines.values():
            sp.set_color(grid)
        ax.tick_params(colors=ink2)
    axes[0].legend(frameon=False, loc="upper right", ncol=3, fontsize=9, labelcolor=ink, bbox_to_anchor=(1, 1.22))
    axes[1].set_xlabel("time (ms)", color=ink2)
    fig.suptitle("nsys GPU metrics (gb10b, 20 kHz) - Jetson AGX Thor", color=ink, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "gpu_metrics.png", dpi=130, facecolor=surface)
    print(f"\n图已保存: {OUT / 'gpu_metrics.png'}")


# ---------------------------------------------------------------- B
def overlap_len(a, b):
    """两组区间（各自内部不重叠）的交集长度。"""
    i = j = tot = 0
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        tot += max(0, e - s)
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return tot


def part_b():
    print("\n=== B. 拷贝与计算：Copy Engine 和 SM 的活动 ===")
    db = nsys("gm_overlap", [str(Path(__file__).resolve()), "--worker-overlap"])
    samples, nvtx, gpu, _ = load(db)
    for name in ("overlap_h2d", "overlap_d2d"):
        lo, hi = next((s, e) for s, e, n in nvtx if n == name)
        mc = [(s, e) for s, e in gpu["memcpy"] if lo <= s < hi]
        kn = [(s, e) for s, e in gpu["kernel"] if lo <= s < hi]
        t_mc, t_kn = sum(e - s for s, e in mc), sum(e - s for s, e in kn)
        ov = overlap_len(mc, kn)
        print(f"\n[{name}] memcpy {len(mc)} 次 合计 {t_mc / 1e6:.2f} ms，GEMM {len(kn)} 次 合计 {t_kn / 1e6:.2f} ms，"
              f"区间重叠 {ov / 1e6:.3f} ms（{100 * ov / max(t_mc, 1):.1f}% 的拷贝时间），墙钟 {(hi - lo) / 1e6:.2f} ms")

        def during(iv):
            out, k = [], 0
            for t, v in samples:
                while k < len(iv) and iv[k][1] <= t:
                    k += 1
                if k < len(iv) and iv[k][0] <= t:
                    out.append(v)
            return out
        for label, iv in (("拷贝期间", mc), ("GEMM 期间", kn)):
            rows = during(iv)
            print(f"  {label:<9s} 样本 {len(rows):4d}  " + "  ".join(f"{SHORT[k]} {mean(rows, k):5.1f}" for k in SHOW))
        print("  前 4 次拷贝/GEMM 的时间（ms，相对区间开始）：")  # noqa
        print("    memcpy: " + "  ".join(f"{(s - lo) / 1e6:.3f}-{(e - lo) / 1e6:.3f}" for s, e in mc[:4]))
        print("    kernel: " + "  ".join(f"{(s - lo) / 1e6:.3f}-{(e - lo) / 1e6:.3f}" for s, e in kn[:4]))


def part_c():
    print("\n=== C. pinned H2D 拷贝由谁执行：Copy Engine 还是 SM？===")
    db = nsys("gm_copy_sizes", [str(Path(__file__).resolve()), "--worker-sizes"])
    samples, nvtx, gpu, _ = load(db)
    for lo, hi, name in nvtx:
        if not name.startswith(("alone_", "withgemm_")):
            continue
        mc = [(s, e) for s, e in gpu["memcpy"] if lo <= s < hi]
        kn = [(s, e) for s, e in gpu["kernel"] if lo <= s < hi]
        size = int(name.split("_")[1][:-2]) * 1024
        rows, k = [], 0
        for t, v in samples:
            while k < len(mc) and mc[k][1] <= t:
                k += 1
            if k < len(mc) and mc[k][0] <= t:
                rows.append(v)
        t_mc = sum(e - s for s, e in mc)
        line = (f"  {name:<17s} {len(mc):5d} 次  每次 {t_mc / len(mc) / 1e3:7.1f} us  {len(mc) * size / t_mc:6.1f} GB/s  拷贝期间: "
                + "  ".join(f"{SHORT[x]} {mean(rows, x):5.1f}" for x in (SHOW[1], SHOW[4], SHOW[5], SHOW[3])))
        if kn:
            t_kn = sum(e - s for s, e in kn)
            line += (f"\n  {'':<17s} GEMM {t_kn / 1e6:.2f} ms + 拷贝 {t_mc / 1e6:.2f} ms = {(t_kn + t_mc) / 1e6:.2f} ms；"
                     f"区间重叠 {overlap_len(mc, kn) / 1e6:.2f} ms；墙钟 {(hi - lo) / 1e6:.2f} ms")
        print(line)


def part_d():
    print("\n=== D. nsys 下 pageable H2D 拷贝的尖峰（观察者效应）===")
    for f in ("pipeline_slow", "gm_slow", "pipeline_fast", "gm_fast"):   # pipeline_* 来自 01_nsys_basics.sh（不开 GPU metrics）
        db = OUT / f"{f}.sqlite"
        if not db.exists():
            continue
        c = sqlite3.connect(db)
        mc = c.execute("SELECT end - start, srcKind FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE copyKind = 1").fetchall()
        d = sorted(x[0] / 1e3 for x in mc)
        kind = {0: "pageable", 1: "pinned"}.get(mc[0][1], mc[0][1])
        print(f"  {f:<14s} H2D {len(d):3d} 次（{kind}）  中位 {d[len(d) // 2]:6.1f} us  超过 1 ms 的 {sum(x > 1000 for x in d):2d} 次  最长 {d[-1]:7.0f} us")
    print("  不开 nsys，直接给慢版流水线的 preprocess 计时（60 步）：")
    import time
    import numpy as np
    import torch
    sys.path.insert(0, str(HERE.parent))
    from common import PipelineConfig, SlowPipeline
    for name, cfg in (("slow", PipelineConfig.slow()), ("slow - no_pin", PipelineConfig.slow(no_pin=False))):
        pipe = SlowPipeline(cfg)
        for i in range(5):
            pipe.step(i)
        pre = []
        for i in range(60):
            f = pipe.capture(i)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            x = pipe.preprocess(f)
            torch.cuda.synchronize(); pre.append((time.perf_counter() - t0) * 1e3)
            pipe.postprocess(pipe.infer(x))
        pre = np.array(pre)
        print(f"    {name:<14s} preprocess 中位 {np.median(pre):.2f} ms  最长 {pre.max():.2f} ms  超过 5 ms 的 {int((pre > 5).sum())} 次")


if __name__ == "__main__":
    if "--worker-overlap" in sys.argv:
        worker_overlap()
        sys.exit()
    if "--worker-sizes" in sys.argv:
        worker_sizes()
        sys.exit()
    log = open(OUT / "07_gpu_metrics.log", "w")

    class Tee:
        def write(self, s): sys.__stdout__.write(s); log.write(s)
        def flush(self): sys.__stdout__.flush(); log.flush()
    sys.stdout = Tee()
    res = part_a()
    plot(res)
    part_b()
    part_c()
    part_d()
