"""5.9 分析慢流水线中的真实 kernel：从 nsys 的"哪些 kernel 占时间"到 ncu 的"这个 kernel 为什么慢"。

    python 07_pipeline_kernels.py

对慢版（fp32）和快版（fp16）流水线各采集一步 infer 阶段的全部 kernel：
    ncu --profile-from-start off   → 只在 cudaProfilerStart/Stop 之间采集（和第 04 章 nsys 的 --capture-range=cudaProfilerApi 是同一套 API）
        --nvtx --nvtx-include "infer/"  → 只要 infer 阶段（不要预处理、后处理的小 kernel）
然后按 kernel 名聚合：耗时占比、SOL、DRAM 利用率、Tensor Core、Occupancy，并对耗时最多的 kernel 打印 ncu 的建议。
"""
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ncu_tools as T  # noqa: E402
from common import PipelineConfig, SlowPipeline  # noqa: E402

EXTRA = [
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_active",
    "launch__grid_size", "launch__block_size",
    "sm__ops_path_tensor_op_utchmma_src_fp16_dst_fp32.sum", "sm__ops_path_tensor_op_hmma_src_fp16_dst_fp32.sum",
    "sm__ops_path_tensor_op_hmma_src_tf32_dst_fp32.sum", "sm__ops_path_tensor_op_utchmma_src_tf32_dst_fp32.sum",
]


def worker():
    fast = "--fast" in sys.argv
    pipe = SlowPipeline(PipelineConfig.fast() if fast else PipelineConfig.slow())
    for i in range(5):
        pipe.step(i)
    torch.cuda.synchronize()
    torch.cuda.profiler.start()          # ncu --profile-from-start off：从这里开始
    pipe.step(5)
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()


def short(name: str) -> str:
    for pre in ("void ", "at::native::", "(anonymous namespace)::"):
        name = name.replace(pre, "")
    return name[:40]


def analyze(tag: str, rows: list[dict]):
    by = defaultdict(list)
    for r in rows:
        by[r["kernel"]].append(r)
    total = sum(r["gpu__time_duration.sum"] for r in rows)
    agg = []
    for k, rs in by.items():
        t = sum(r["gpu__time_duration.sum"] for r in rs)
        main_r = max(rs, key=lambda r: r["gpu__time_duration.sum"])
        agg.append(dict(main_r, variant=short(k), n=len(rs), t_us=t / 1e3, share=100 * t / total,
                        avg_us=t / len(rs) / 1e3,
                        dram=100 * sum(T.sysmem_bytes(r) for r in rs) / t / T.PEAK_BW_GBS))
    agg.sort(key=lambda r: -r["t_us"])
    print(f"\n[{tag}] infer 阶段共 {len(rows)} 个 kernel、{len(by)} 种，ncu 测得的 GPU 时间合计 {total / 1e3:.1f} us")
    T.table(agg[:12], [("kernel", "variant", ""), ("n", "n", "d"), ("total_us", "t_us", ".1f"), ("share%", "share", ".1f"),
                       ("avg_us", "avg_us", ".1f"), ("grid", "launch__grid_size", ".0f"), ("block", "launch__block_size", ".0f"),
                       T.COL_SM, ("DRAM%", "dram", ".0f"),
                       ("tensor%", "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active", ".0f"),
                       ("tc%", "sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_active", ".0f"),
                       T.COL_OCC, T.COL_REGS],
            title=f"== {tag}：按 kernel 名聚合，按总耗时排序（前 12）；SM%/tensor%/Occ% 取该名字下耗时最长的一次 ==")
    small = [r for r in rows if r["gpu__time_duration.sum"] < 5e3]
    print(f"  短于 5 us 的 kernel：{len(small)} 个（{100 * len(small) / len(rows):.0f}%），"
          f"合计 {sum(r['gpu__time_duration.sum'] for r in small) / 1e3:.1f} us（{100 * sum(r['gpu__time_duration.sum'] for r in small) / total:.0f}%）")
    for r in agg[:2]:
        print(f"  ncu 对 {r['variant']} 的建议：")
        for s in T.rules(r, min_speedup=10)[:3]:
            print("    -", s)
    return agg


def main():
    if T.worker_mode():
        worker()
        return
    T.tee("07_pipeline_kernels")
    res = {}
    for tag, args in (("slow_fp32", ()), ("fast_fp16", ("--fast",))):
        print(f"\n== ncu 采集 {tag} ==")
        rep = T.profile(__file__, f"07_{tag}", sections=["SpeedOfLight", "Occupancy", "LaunchStats", "WarpStateStats"],
                        metrics=T.BASE_METRICS + EXTRA, nvtx_include="infer/",
                        extra=["--profile-from-start", "off"], script_args=args)
        res[tag] = analyze(tag, T.load(rep))


if __name__ == "__main__":
    main()
