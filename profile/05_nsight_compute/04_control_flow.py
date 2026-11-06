"""5.6 Control-flow-bound：warp divergence。

    python 04_control_flow.py

A. if/else 分支（N = 4M，每个分支 512 次 FMA，两个分支的计算量相同）
    A_divergent      if (threadIdx.x & 1)        → 每个 warp 内一半线程走 A、一半走 B，两条路径串行执行
    A_warp_uniform   if ((threadIdx.x >> 5) & 1) → 以 warp 为单位分支，总工作量完全一样

B. 数据相关的循环次数（每个元素迭代 cnt[i] 次，cnt ~ U[0, 1024)）
    B_random         原始顺序：warp 的耗时 = 32 个线程中 cnt 的最大值
    B_sorted         先按 cnt 排序再处理（同一个 kernel，只是输入顺序变了）→ 同一个 warp 内的 cnt 接近
    B_uniform_mean   参照：所有元素都迭代平均次数（总 FMA 数相同、完全没有 divergence）
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ncu_tools as T  # noqa: E402
from kernels import K  # noqa: E402

N = 4 * 1024 * 1024
ITERS = 512
xa = torch.rand(N, device="cuda")

g = torch.Generator(device="cuda").manual_seed(0)
cnt_random = torch.randint(0, 1024, (N,), device="cuda", generator=g, dtype=torch.int32)
cnt_sorted, order = torch.sort(cnt_random)           # 排序本身的开销见 README（这里不计入 kernel 时间）
cnt_sorted = cnt_sorted.contiguous()
cnt_mean = torch.full((N,), int(cnt_random.float().mean().round()), device="cuda", dtype=torch.int32)
xb = torch.rand(N, device="cuda")
xb_sorted = xb[order].contiguous()

VARIANTS = {
    "A_divergent": lambda: K.branch(0, xa, ITERS),
    "A_warp_uniform": lambda: K.branch(1, xa, ITERS),
    "B_random": lambda: K.loop_varcount(xb, cnt_random),
    "B_sorted": lambda: K.loop_varcount(xb_sorted, cnt_sorted),
    "B_uniform_mean": lambda: K.loop_varcount(xb, cnt_mean),
}

EXTRA = [
    "smsp__thread_inst_executed_per_inst_executed.ratio",          # Avg. Active Threads Per Warp（满分 32）
    "smsp__sass_average_branch_targets_threads_uniform.pct",       # Branch Efficiency：所有活跃线程走同一方向的分支占比
    "sm__sass_branch_targets_threads_divergent.sum",               # 发生分歧的分支次数
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",         # 线程级 FFMA 数（真正有用的计算）
    "smsp__inst_executed_pipe_fma.sum",                            # warp 级 FMA 指令数
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active",
]


def main():
    if T.worker_mode():
        T.run_variants(VARIANTS, warmup=1)
        return
    T.tee("04_control_flow")
    print(f"B: cnt 均值 {cnt_random.float().mean():.1f}，每个 warp 内 cnt 最大值的均值 "
          f"{cnt_random.view(-1, 32).amax(1).float().mean():.1f}（随机顺序）vs "
          f"{cnt_sorted.view(-1, 32).amax(1).float().mean():.1f}（排序后）")
    t_sort = T.gpu_time_ms(lambda: torch.sort(cnt_random), flush=False)
    print(f"   排序（torch.sort {N/2**20:.0f}M 个 int32）本身要 {t_sort:.3f} ms\n")

    print("== 1. 计时（CUDA Event，中位数；计算密集，不刷 L2）==")
    ms = {k: T.gpu_time_ms(f, flush=False) for k, f in VARIANTS.items()}
    base = {"A": ms["A_divergent"], "B": ms["B_random"]}
    for k, v in ms.items():
        print(f"  {k:<16s} {v:7.3f} ms  ({base[k[0]] / v:.2f}x)")

    print("\n== 2. ncu 采集 ==")
    rep = T.profile(__file__, "04_control_flow",
                    sections=["SpeedOfLight", "ComputeWorkloadAnalysis", "WarpStateStats", "InstructionStats",
                              "SchedulerStats", "SourceCounters"],
                    metrics=T.BASE_METRICS + EXTRA)
    rows = T.load(rep)
    for r in rows:
        r["kernel"] = r["kernel"][:20]
    T.table(rows, [T.COL_VARIANT, ("kernel", "kernel", ""), T.COL_TIME, T.COL_SM, T.COL_IPC,
                   ("issue%", "smsp__issue_active.avg.pct_of_peak_sustained_active", ".0f"),
                   ("FMA_pipe%", "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active", ".0f"),
                   ("active_thr/warp", "smsp__thread_inst_executed_per_inst_executed.ratio", ".1f"),
                   ("branch_eff%", "smsp__sass_average_branch_targets_threads_uniform.pct", ".1f"),
                   ("div_branches(M)", lambda r: r["sm__sass_branch_targets_threads_divergent.sum"] / 1e6, ".2f"),
                   ("warp_FMA(M)", lambda r: r["smsp__inst_executed_pipe_fma.sum"] / 1e6, ".0f"),
                   ("thread_FMA(G)", lambda r: r["smsp__sass_thread_inst_executed_op_ffma_pred_on.sum"] / 1e9, ".2f"),
                   ("FMA有效率%", lambda r: 100 * r["smsp__sass_thread_inst_executed_op_ffma_pred_on.sum"]
                    / (32 * r["smsp__inst_executed_pipe_fma.sum"]), ".0f")],
            title="== 3. divergence 的特征：活跃线程数、分支效率、warp 指令数 vs 线程指令数 ==")

    for v in ("A_divergent", "B_random"):
        print(f"\n== 4. ncu 规则引擎的建议（{v}）==")
        for s in T.rules(next(r for r in rows if r["variant"] == v), min_speedup=1)[:4]:
            print("  -", s)


if __name__ == "__main__":
    main()
