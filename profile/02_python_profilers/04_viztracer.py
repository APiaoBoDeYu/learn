"""02-04 VizTracer：确定性 tracing，记录每一次函数调用的起止时间，输出时间线（Perfetto 可以打开）。

演示内容：
  A. 基本用法：追踪慢流水线 20 步 → outputs/viztracer_slow.json
  B. 控制开销：include_files / max_stack_depth / ignore_c_function / min_duration（以及 include_files 的剪枝坑）
  C. 自定义事件：log_event（区间）、log_instant（瞬时）、add_counter（数值曲线）
  D. 异步陷阱：GPU 时间被算到了哪里？用 CUDA_LAUNCH_BLOCKING=1 对比

运行：/opt/venv/bin/python 02_python_profilers/04_viztracer.py
查看：把 outputs/*.json 拖进 https://ui.perfetto.dev ，或者运行 vizviewer outputs/viztracer_slow.json
"""
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from viztracer import VizTracer  # noqa: E402

from workload import build, run  # noqa: E402

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)
STAGES = ["capture", "preprocess", "infer", "postprocess"]


def stage_times(path: Path, steps: int) -> dict:
    """从 trace 中统计流水线各阶段函数的平均耗时（ms/step）。"""
    ev = json.load(open(path))["traceEvents"]
    acc = defaultdict(float)
    for e in ev:
        if e.get("ph") != "X":
            continue
        name = e["name"].split(" (")[0]          # "SlowPipeline.infer (.../pipeline.py:130)" -> "SlowPipeline.infer"
        short = name.split(".")[-1]
        if short in STAGES and name.startswith("SlowPipeline"):
            acc[short] += e["dur"] / 1000
    return {k: acc[k] / steps for k in STAGES}


def n_events(path: Path) -> int:
    return sum(1 for e in json.load(open(path))["traceEvents"] if e.get("ph") == "X")


def main():
    steps = 20
    pipe = build(fast=False)
    base = run(pipe, steps)
    print(f"不开 tracer:                      {base:7.2f} ms/step")

    # ---------- A. 默认设置：追踪所有 Python 函数调用 ----------
    tracer = VizTracer(output_file=str(OUT / "viztracer_slow.json"), verbose=0)
    tracer.start()
    ms_full = run(pipe, steps)
    tracer.stop()
    tracer.save()
    print(f"A 默认（全部追踪）:                {ms_full:7.2f} ms/step  "
          f"开销 {ms_full / base - 1:+.0%}  事件数 {n_events(OUT / 'viztracer_slow.json')}")

    # ---------- B. 控制开销 ----------
    # 坑：include_files 会剪掉"不在列表里的函数"的整棵子树。
    #   1) run() 在本目录的 workload.py 里：如果只写 common/，run() 被剪掉 → 事件数为 0
    #   2) infer/postprocess 被 @torch.no_grad() 装饰，调用链是 step → decorate_context(torch 内部) → infer，
    #      decorate_context 不在列表里 → infer/postprocess 整棵子树被剪掉
    common_dir = str(Path(__file__).resolve().parents[1] / "common") + "/"
    here_dir = str(Path(__file__).resolve().parent) + "/"
    variants = [
        ("B1 include=[common/]", dict(include_files=[common_dir])),
        ("B2 include=[common/, 本目录]", dict(include_files=[common_dir, here_dir])),
        ("B3 depth<=6 + ignore_c", dict(max_stack_depth=6, ignore_c_function=True)),
        ("B4 depth<=6 + min_dur 20us", dict(max_stack_depth=6, min_duration=20)),
    ]
    for name, kw in variants:
        path = OUT / f"viztracer_{name.split()[0]}.json"
        tracer = VizTracer(output_file=str(path), verbose=0, **kw)
        tracer.start()
        ms = run(pipe, steps)
        tracer.stop()
        tracer.save()
        st = stage_times(path, steps)
        seen = [k for k in STAGES if st[k] > 0]
        print(f"{name:<32s} {ms:7.2f} ms/step  开销 {ms / base - 1:+4.0%}  事件数 {n_events(path):6d}  "
              f"记录到的阶段 {seen}")

    # ---------- C. 自定义事件 ----------
    tracer = VizTracer(output_file=str(OUT / "viztracer_custom.json"), verbose=0, max_stack_depth=8)
    tracer.start()
    for i in range(steps):
        with tracer.log_event(f"frame {i}"):           # 自定义区间：每帧一个
            pipe.step(i)
        tracer.add_counter("pipeline_state", {         # 数值曲线：在 Perfetto 中显示为 counter 轨道
            "n_warnings": pipe.n_warnings,
            "cuda_mem_mb": torch.cuda.memory_allocated() / 2 ** 20})
        if i % 5 == 0:
            tracer.log_instant("checkpoint", args={"step": i})   # 瞬时事件
    torch.cuda.synchronize()
    tracer.stop()
    tracer.save()
    print("C 自定义事件: outputs/viztracer_custom.json（frame 区间 + pipeline_state 计数器 + checkpoint 标记）")

    # ---------- D. 异步陷阱 ----------
    print("\n=== D. 各阶段的 CPU 侧耗时（ms/step），来自 trace 统计 ===")
    tracer = VizTracer(output_file=str(OUT / "viztracer_async.json"), verbose=0, max_stack_depth=8)
    tracer.start()
    run(pipe, steps)
    tracer.stop()
    tracer.save()
    a = stage_times(OUT / "viztracer_async.json", steps)

    # 对照组：CUDA_LAUNCH_BLOCKING=1 让每个 kernel launch 都同步等待执行完成，
    # 这样 GPU 的执行时间会"留在"发起它的那个函数里。必须在 CUDA 初始化之前设置，所以放在子进程里跑。
    env = dict(os.environ, CUDA_LAUNCH_BLOCKING="1")
    subprocess.run([sys.executable, __file__, "--blocking-child", str(steps)], env=env, check=True)
    b = stage_times(OUT / "viztracer_blocking.json", steps)
    print(f"{'stage':<14s}{'默认(异步)':>12s}{'LAUNCH_BLOCKING':>17s}")
    for k in STAGES:
        print(f"{k:<14s}{a[k]:12.2f}{b[k]:17.2f}")
    print(f"{'sum':<14s}{sum(a.values()):12.2f}{sum(b.values()):17.2f}")


def blocking_child(steps: int):
    pipe = build(fast=False)
    tracer = VizTracer(output_file=str(OUT / "viztracer_blocking.json"), verbose=0, max_stack_depth=8)
    tracer.start()
    ms = run(pipe, steps)
    tracer.stop()
    tracer.save()
    print(f"[子进程 CUDA_LAUNCH_BLOCKING=1] {ms:.2f} ms/step")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--blocking-child":
        blocking_child(int(sys.argv[2]))
    else:
        main()
