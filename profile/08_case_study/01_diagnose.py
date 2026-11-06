"""08-01 按漏斗模型诊断慢流水线（S0 基线）

    /opt/venv/bin/python 08_case_study/01_diagnose.py

L0 系统级    jtop：GPU 负载、CPU 最忙核、功耗（第 01 章）
L1 Python 级  cProfile：tottime 前 8（第 02 章）
L2 框架级    torch.profiler（只开 CPU 活动；Thor 上拿不到 GPU 时间，第 03 章）：算子调用次数、同步
L3 时间线    nsys + SQL：各阶段 CPU 时间、GPU busy、kernel 数、短 kernel 比例（第 04 章）
L4 kernel 级 ncu：引用第 05 章 5.9 的结果（采集 141 个 kernel 要约 5 分钟，这里不重复）
"""
import cProfile
import importlib.util
import io
import pstats
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import case  # noqa: E402
from common import PipelineConfig, SlowPipeline  # noqa: E402

lines = []


def p(s=""):
    print(s); lines.append(s)


def l0(pipe):
    r = case.power_during(pipe, 4.0)
    p("== L0 系统级（jtop，连续运行 4 s）==")
    p(f"  {r['ms_per_step']:.2f} ms/step；GPU 负载 {r['gpu_load']:.0f}%；最忙的 CPU 核 {r['cpu_max']:.0f}%；"
      f"总功耗 {r['p_tot_w']:.1f} W（GPU {r['p_gpu_w']:.1f} W）")
    p("  → 判断：GPU 没有吃满、CPU 有一个核接近满载 → 先看 CPU 侧（L1）")


def l1(pipe):
    p("\n== L1 Python 级（cProfile，50 步，按 tottime 排序）==")
    pr = cProfile.Profile()
    pr.enable()
    for i in range(50):
        pipe.step(i)
    torch.cuda.synchronize()
    pr.disable()
    s = io.StringIO()
    st = pstats.Stats(pr, stream=s).sort_stats("tottime")
    tot = st.total_tt
    rows = sorted(st.stats.items(), key=lambda kv: -kv[1][2])[:8]
    for (fn, line, name), (cc, nc, tt, ct, _) in rows:
        p(f"  {100 * tt / tot:5.1f}%  {tt / 50 * 1000:6.3f} ms/step  {nc // 50:6d} 次/step  {Path(fn).name}:{line}({name})")
    p("  （cProfile 本身让程序变慢约 40%，这里只看相对比例，第 02 章）")


def l2(pipe):
    from torch.profiler import ProfilerActivity, profile
    p("\n== L2 框架级（torch.profiler，只开 CPU，20 步）==")
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        for i in range(20):
            pipe.step(i)
        torch.cuda.synchronize()
    ka = sorted(prof.key_averages(), key=lambda e: -e.self_cpu_time_total)
    tot = sum(e.self_cpu_time_total for e in ka)
    n_ops = sum(e.count for e in ka if e.key.startswith("aten::"))
    p(f"  每步 {n_ops // 20} 次 aten 调用；self CPU 时间前 8：")
    for e in ka[:8]:
        p(f"  {100 * e.self_cpu_time_total / tot:5.1f}%  {e.key:<34s} {e.count // 20:5d} 次/step  {e.self_cpu_time_total / 20 / 1000:6.3f} ms/step")
    item = [e for e in ka if e.key in ("aten::item", "aten::_local_scalar_dense")]
    p("  同步相关：" + "，".join(f"{e.key} {e.count // 20} 次/step、{e.cpu_time_total / 20 / 1000:.3f} ms/step" for e in item))


def l3():
    p("\n== L3 时间线（nsys，20 步，-t cuda,nvtx,osrt）==")
    root = case.ROOT
    rep = case.OUT / "s0_nsys"
    subprocess.run(["nsys", "profile", "-t", "cuda,nvtx,osrt", "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
                    "-o", str(rep), "--force-overwrite", "true", sys.executable, str(root / "04_nsight_systems" / "workload_nvtx.py"),
                    "--steps", "20"], capture_output=True)
    subprocess.run(["nsys", "export", "-t", "sqlite", "-o", f"{rep}.sqlite", "--force-overwrite", "true", f"{rep}.nsys-rep"],
                   capture_output=True)
    spec = importlib.util.spec_from_file_location("an", root / "04_nsight_systems" / "02_analyze_sqlite.py")
    an = importlib.util.module_from_spec(spec); spec.loader.exec_module(an)
    nvtx, gpu, _ = an.load(Path(f"{rep}.sqlite"))
    steps = [(s, e) for s, e, n in nvtx if n and n.startswith("step_")]
    lo, hi = steps[0][0], steps[-1][1]
    busy = an.union_len([(s, e) for s, e, _, _ in gpu], lo, hi)
    p(f"  {(hi - lo) / len(steps) / 1e6:.2f} ms/step（nsys 下，比不开时慢约 30%）；GPU busy {busy / (hi - lo):.1%}")
    p(f"  {'阶段':<12s} {'CPU ms/step':>11s} {'GPU ms/step':>11s} {'kernel/step':>11s} {'<5us 的比例':>11s}")
    for ph in ("preprocess", "infer", "postprocess"):
        rs = [(s, e) for s, e, n in nvtx if n == ph and lo <= s < hi]
        cpu = sum(e - s for s, e in rs)
        ks = [(s, e) for s, e, kind, _ in gpu if kind == "kernel" and any(a <= s < b for a, b in rs)]
        g = sum(an.union_len([(s, e) for s, e, _, _ in gpu], a, b) for a, b in rs)
        short = sum(1 for s, e in ks if e - s < 5000)
        p(f"  {ph:<12s} {cpu / len(steps) / 1e6:11.3f} {g / len(steps) / 1e6:11.3f} {len(ks) / len(steps):11.0f} "
          f"{100 * short / max(len(ks), 1):10.0f}%")


def l4():
    p("\n== L4 kernel 级（ncu，引用第 05 章 5.9：慢版 infer 阶段 141 个 kernel）==")
    p("  Kernel2（cuBLAS fp32 GEMM）占 47.7%：198 regs、理论 Occupancy 16.7%、grid 64 → Tail Effect")
    p("  nchwToNhwcKernel 占 8.5%（fp32 卷积的布局转换）")
    p("  → fp16 + channels_last 能同时解决这两项")


def main():
    torch.backends.mha.set_fastpath_enabled(True)
    pipe = SlowPipeline(PipelineConfig.slow())
    for i in range(10):
        pipe.step(i)
    l0(pipe); l1(pipe); l2(pipe); l3(); l4()
    (case.OUT / "01_diagnose.log").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
