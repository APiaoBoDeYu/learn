"""04-06 用 nsys 得到"每个 PyTorch 算子的 GPU 时间"——替代 Thor 上 torch.profiler 缺失的 CUDA time 列。

原理：
  1. nsys profile --pytorch=autograd-nvtx 会为每个 aten 算子生成 NVTX range（名字形如 "aten::linear, op_id = 123"）
  2. 每个 kernel 通过 correlationId 找到发起它的 cudaLaunchKernel 调用（CPU 侧，有时间戳和线程号）
  3. 包住这次 launch 的**最内层** aten:: range，就是发起这个 kernel 的算子
  4. 按算子名聚合：kernel 数、GPU 时间、CPU 时间

运行：
    nsys profile -t cuda,nvtx --pytorch=autograd-nvtx --capture-range=cudaProfilerApi --capture-range-end=stop \\
        -o 04_nsight_systems/outputs/pt_ops --force-overwrite true \\
        /opt/venv/bin/python 04_nsight_systems/workload_nvtx.py --steps 10 [--fast]
    nsys export -t sqlite -o 04_nsight_systems/outputs/pt_ops.sqlite 04_nsight_systems/outputs/pt_ops.nsys-rep
    /opt/venv/bin/python 04_nsight_systems/06_op_gpu_time.py 04_nsight_systems/outputs/pt_ops.sqlite
"""
import bisect
import re
import sqlite3
import sys
from collections import defaultdict


def main():
    db = sys.argv[1] if len(sys.argv) > 1 else "04_nsight_systems/outputs/pt_ops.sqlite"
    top = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    from pathlib import Path
    if not Path(db).exists():      # sqlite3.connect 会悄悄新建一个空库，之后报 "no such table"，所以先检查
        sys.exit(f"找不到 {db}：请先按本文件开头的命令用 nsys --pytorch=autograd-nvtx 采集并导出 sqlite")
    c = sqlite3.connect(db)
    rows = c.execute("""SELECT start, end, globalTid, COALESCE(text, (SELECT value FROM StringIds WHERE id = textId))
                        FROM NVTX_EVENTS WHERE eventType = 59 AND end IS NOT NULL""").fetchall()
    # --pytorch=autograd-nvtx 会给 record_function 再生成一个带 op_id 的同名 range，只数不带逗号的那份
    n_steps = sum(1 for r in rows if r[3] and r[3].startswith(("step_", "settle_")) and "," not in r[3])
    ops = defaultdict(list)          # 线程 → [(start, end, name)]，只保留 aten 算子
    for s, e, tid, name in rows:
        if name and name.startswith("aten::"):
            ops[tid].append((s, e, re.split(r",\s*(op_id|sizes|seq)", name)[0]))
    for v in ops.values():
        v.sort()
    starts = {tid: [r[0] for r in v] for tid, v in ops.items()}

    launch = {cid: (s, tid) for cid, s, tid in c.execute(
        "SELECT correlationId, start, globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME")}
    kernels = c.execute("SELECT start, end, correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchall()

    gpu = defaultdict(float)
    nk = defaultdict(int)
    unmatched = 0
    for ks, ke, cid in kernels:
        if cid not in launch:
            unmatched += 1
            continue
        t, tid = launch[cid]
        rs, st = ops.get(tid, []), starts.get(tid, [])
        i = bisect.bisect_right(st, t) - 1
        owner = None
        while i >= 0:                       # 从开始时间最晚的 range 往回找，第一个包住 t 的就是最内层
            s, e, name = rs[i]
            if e >= t:
                owner = name
                break
            i -= 1
        key = owner or "(不在任何 aten 算子内)"
        gpu[key] += ke - ks
        nk[key] += 1

    # CPU 时间：同名算子的 range 时长之和（嵌套算子会重复计算，所以只作参考）
    cpu = defaultdict(float)
    cnt = defaultdict(int)
    for v in ops.values():
        for s, e, name in v:
            cpu[name] += e - s
            cnt[name] += 1

    total = sum(gpu.values())
    steps = max(n_steps, 1)
    print(f"{len(kernels)} 个 kernel，{n_steps} 个 step，GPU kernel 时间合计 {total / steps / 1e6:.3f} ms/step"
          + (f"，{unmatched} 个未匹配到 launch" if unmatched else ""))
    print(f"\n{'算子（发起 kernel 的最内层 aten 算子）':<40s}{'调用/step':>10s}{'kernel/step':>12s}"
          f"{'GPU ms/step':>12s}{'GPU%':>7s}{'CPU ms/step':>12s}")
    for name, g in sorted(gpu.items(), key=lambda x: -x[1])[:top]:
        print(f"{name:<44s}{cnt.get(name, 0) / steps:8.1f}{nk[name] / steps:12.1f}{g / steps / 1e6:12.3f}"
              f"{g / total:7.1%}{cpu.get(name, 0) / steps / 1e6:12.3f}")


if __name__ == "__main__":
    main()
