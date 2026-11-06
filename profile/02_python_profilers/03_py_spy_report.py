"""02-03 汇总 py-spy 的 raw（折叠栈）输出，得到"文本版火焰图"。

raw 格式每行是：  frame1;frame2;...;leaf  样本数
  - self（自身）：leaf 是该函数的样本数，即 CPU 时间花在这个函数本身
  - total（累计）：栈中任意位置出现该函数的样本数，包含它调用的子函数

用法：python 03_py_spy_report.py stacks.txt [--focus 函数名] [--grep 关键字] [--top N]
  --focus：只统计包含该函数的调用栈，并从该函数处截断（去掉 import torch、模型初始化等启动阶段的样本）
"""
import argparse
import re
from collections import Counter


def short(frame: str) -> str:
    # "postprocess (common/pipeline.py:150)" -> "postprocess (pipeline.py:150)"
    return re.sub(r"\(([^()]*/)?([^/()]+)\)$", r"(\2)", frame.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--grep", default=None, help="只显示包含该关键字的调用栈")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--focus", default="step (pipeline.py", help="只看包含该帧的栈（默认：流水线的 step）")
    args = ap.parse_args()

    self_c, total_c, stacks = Counter(), Counter(), Counter()
    n = n_all = 0
    for line in open(args.path):
        line = line.rstrip()
        if not line:
            continue
        stack, cnt = line.rsplit(" ", 1)
        cnt = int(cnt)
        frames = [short(f) for f in stack.split(";")]
        n_all += cnt
        if args.focus:
            idx = [k for k, f in enumerate(frames) if f.startswith(args.focus)]
            if not idx:
                continue
            frames = frames[idx[0]:]
        n += cnt
        self_c[frames[-1]] += cnt
        for f in {re.sub(r":\d+\)$", ")", f) for f in frames}:  # 累计按函数聚合，不区分行号
            total_c[f] += cnt
        stacks[tuple(frames)] += cnt

    if args.grep:
        print(f"包含 '{args.grep}' 的调用栈（共 {n} 个样本）：")
        hits = [(s, c) for s, c in stacks.items() if any(args.grep in f for f in s)]
        for s, c in sorted(hits, key=lambda x: -x[1])[:3]:
            i = next(k for k, f in enumerate(s) if args.grep in f)
            print(f"\n  {c / n:6.1%}  " + "\n          -> ".join(s[max(i - 2, 0):]))
        return

    print(f"总样本数 {n_all}，其中 '{args.focus}' 内的稳态样本 {n}（{n / n_all:.0%}），以下百分比均相对于稳态样本")
    print(f"\n{'self%':>7s}  function（自身耗时最多）")
    for f, c in self_c.most_common(args.top):
        print(f"{c / n:7.1%}  {f}")
    print(f"\n{'total%':>7s}  function（累计耗时，含子调用，按函数聚合）")
    for f, c in total_c.most_common(args.top):
        print(f"{c / n:7.1%}  {f}")


if __name__ == "__main__":
    main()
