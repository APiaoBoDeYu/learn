"""02-02 line_profiler：逐行耗时。cProfile 找到热点函数之后，用它看函数内部是哪一行慢。

运行：/opt/venv/bin/python 02_python_profilers/02_line_profiler.py

命令行用法（需要在函数上加 @profile 装饰器）：
    kernprof -l -v script.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from line_profiler import LineProfiler  # noqa: E402

from workload import build, run  # noqa: E402


def main():
    pipe = build(fast=False)
    lp = LineProfiler()
    # 不用改源码：直接把要分析的函数注册进去（cProfile 找出来的两个热点）
    lp.add_function(type(pipe)._preprocess_python)
    lp.add_function(type(pipe).postprocess.__wrapped__)   # postprocess 被 @torch.no_grad() 包了一层
    pipe.postprocess = lp(type(pipe).postprocess.__wrapped__).__get__(pipe)
    pipe._preprocess_python = lp(type(pipe)._preprocess_python).__get__(pipe)
    ms = run(pipe, 30)
    print(f"慢流水线 30 步（开着 line_profiler）: {ms:.2f} ms/step\n")
    lp.print_stats(output_unit=1e-3, stripzeros=True)


if __name__ == "__main__":
    main()
