"""第 02 章的公共负载：运行慢流水线 N 步。

本章所有工具分析的都是这个脚本，方便比较不同工具的视角：
    /opt/venv/bin/python 02_python_profilers/workload.py --steps 50          # 慢流水线（全部低效开关打开）
    /opt/venv/bin/python 02_python_profilers/workload.py --steps 50 --fast   # 优化版
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common import PipelineConfig, SlowPipeline  # noqa: E402


def allow_ptrace():
    """yama ptrace_scope=1 时，只有父进程能 ptrace 子进程。本进程主动声明允许任何进程追踪自己。"""
    import ctypes
    PR_SET_PTRACER, PR_SET_PTRACER_ANY = 0x59616D61, ctypes.c_ulong(-1)
    ctypes.CDLL(None).prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0)


def build(fast: bool = False) -> SlowPipeline:
    pipe = SlowPipeline(PipelineConfig.fast() if fast else PipelineConfig.slow())
    for i in range(5):  # warmup：cuDNN 选算法、CUDA context 初始化等，不计入分析
        pipe.step(i)
    torch.cuda.synchronize()
    return pipe


def run(pipe: SlowPipeline, steps: int) -> float:
    t0 = time.perf_counter()
    for i in range(steps):
        pipe.step(i)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / steps * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--allow-ptrace", action="store_true",
                    help="允许任意进程 ptrace 本进程（用于 py-spy dump/top --pid，见 README 2.3）")
    args = ap.parse_args()
    if args.allow_ptrace:
        allow_ptrace()
    pipe = build(args.fast)
    ms = run(pipe, args.steps)
    print(f"[{'fast' if args.fast else 'slow'}] {args.steps} steps, {ms:.3f} ms/step")


if __name__ == "__main__":
    main()
