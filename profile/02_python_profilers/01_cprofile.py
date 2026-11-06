"""02-01 cProfile：Python 标准库自带的确定性 profiler（函数级）。

运行：/opt/venv/bin/python 02_python_profilers/01_cprofile.py
产物：outputs/pipeline.prof（可以用 snakeviz 可视化）

也可以不改代码，直接用命令行：
    /opt/venv/bin/python -m cProfile -o outputs/pipeline.prof 02_python_profilers/workload.py --steps 50
"""
import cProfile
import io
import pstats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workload import build, run  # noqa: E402

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)


def show(stats: pstats.Stats, key: str, n: int, title: str):
    buf = io.StringIO()
    stats.stream = buf
    stats.sort_stats(key).print_stats(n)
    text = buf.getvalue()
    # 只保留表格部分，去掉开头的汇总行
    body = text[text.index("   ncalls"):]
    print(f"\n=== {title}（按 {key} 排序，前 {n} 行）===")
    print(body.replace(str(Path(__file__).resolve().parents[1]) + "/", ""))


def main():
    pipe = build(fast=False)
    prof = cProfile.Profile()
    prof.enable()
    ms = run(pipe, 50)
    prof.disable()
    print(f"慢流水线 50 步（开着 cProfile）: {ms:.2f} ms/step")
    prof.dump_stats(OUT / "pipeline.prof")

    st = pstats.Stats(prof)
    st.strip_dirs()
    show(st, "tottime", 12, "函数自身耗时 tottime")
    show(st, "cumulative", 12, "累计耗时 cumtime（包括子调用）")

    # 定向查看：谁调用了 item()？
    print("\n=== 谁调用了 Tensor.item（print_callers）===")
    buf = io.StringIO()
    st.stream = buf
    st.print_callers("item")
    print(buf.getvalue().split("Function was called by...")[-1].strip()[:800])


if __name__ == "__main__":
    main()
