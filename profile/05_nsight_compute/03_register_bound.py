"""5.5 Register-bound：寄存器 → Occupancy → 延迟隐藏。Occupancy 低到底是不是问题？

    python 03_register_bound.py

A. 批量 8x8 小矩阵乘法 C = A·B（batch = 1M，SoA 布局，A/B/C 各 256 MB）—— 每个线程内部 ILP 很高
    A_per_thread       一个线程算一整个矩阵：a[64]、b[64]、c[64] 全在寄存器里（128 regs）
    A_per_thread_r64   同样的代码 + __maxnreg__(64)（相当于 -maxrregcount=64）→ spill 到 local memory
    A_row_per_thread   一个线程算一行（40 regs，Occupancy 100%）
    A_torch_bmm        参照：torch.bmm（AoS 布局 [batch, 8, 8]）

B. 滤波器组：每个元素 128 个状态，沿时间维 T=256 步更新 s[j] = s[j]*w_j + x_t —— 每轮只有 1 个 load，ILP 低
    B_per_thread       一个线程 128 个状态（168 regs，Occupancy 25%）
    B_split4           4 个线程分担（48 regs，Occupancy 高），但每个 warp 每轮的计算量只有 1/4
    B_prefetch         同 B_per_thread，但提前 4 轮发出 load（软件预取，180 regs，Occupancy 仍是 25%）
    B_split4_prefetch  两者结合
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kernels  # noqa: E402
import ncu_tools as T  # noqa: E402
from kernels import K  # noqa: E402

# ---- A ----
BATCH, D = 1 << 20, 8
A = torch.randn(D * D, BATCH, device="cuda")       # SoA：A[e][m]
B = torch.randn(D * D, BATCH, device="cuda")
C = torch.empty_like(A)
A_aos = A.t().reshape(BATCH, D, D).contiguous()
B_aos = B.t().reshape(BATCH, D, D).contiguous()
C_aos = torch.empty_like(A_aos)
# ---- B ----
TSTEPS, NF, S = 256, 1 << 18, 128
X = torch.randn(TSTEPS, NF, device="cuda")
OUT = torch.empty(S, NF, device="cuda")

VARIANTS = {
    "A_per_thread": lambda: K.bmm8(0, A, B, C, block=128),
    "A_per_thread_r64": lambda: K.bmm8(1, A, B, C, block=128),
    "A_row_per_thread": lambda: K.bmm8(2, A, B, C),
    "A_torch_bmm": lambda: torch.bmm(A_aos, B_aos, out=C_aos),
    "B_per_thread": lambda: K.filter128(0, X, OUT),
    "B_split4": lambda: K.filter128(1, X, OUT),
    "B_prefetch": lambda: K.filter128(2, X, OUT),
    "B_split4_prefetch": lambda: K.filter128(3, X, OUT),
}
WORK = {  # (FLOP, 必需字节)
    "A": (2 * D ** 3 * BATCH, 3 * D * D * BATCH * 4),
    "B": (2 * S * TSTEPS * NF, (TSTEPS + S) * NF * 4),
}

EXTRA = [
    "launch__occupancy_limit_registers", "launch__occupancy_limit_warps", "launch__block_size",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum", "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum",
    "smsp__warps_active.avg.per_cycle_active",                 # 每个 scheduler 上平均驻留的 warp
    "smsp__warps_eligible.avg.per_cycle_active",               # 每周期"已就绪、可发射"的 warp
    "smsp__issue_active.avg.pct_of_peak_sustained_active",     # Issue Slot 利用率
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio",
    "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active",
]


def check():
    ref = torch.bmm(A_aos, B_aos)
    for v in (0, 1, 2):
        C.zero_(); K.bmm8(v, A, B, C)
        assert torch.allclose(C.t().reshape(BATCH, D, D), ref, atol=1e-3, rtol=1e-3), v
    K.filter128(0, X, OUT); ref = OUT.clone()
    for v in (1, 2, 3):
        OUT.zero_(); K.filter128(v, X, OUT)
        assert torch.allclose(OUT, ref, atol=1e-4), v


def main():
    if T.worker_mode():
        T.run_variants(VARIANTS)
        return
    T.tee("03_register_bound")
    check()
    print("== 0. 编译期就能看到的寄存器 / spill（nvcc --resource-usage）==")
    for name, r in kernels.resource_table().items():
        if name.startswith(("bmm8", "filter128")):
            print(f"  {name:<26s} regs={r['regs']:<4d} spill_stores={r['spill_st']:<5d}B spill_loads={r['spill_ld']:<5d}B")

    print("\n== 1. 计时（CUDA Event + 刷 L2，中位数）==")
    for part in "AB":
        flop, nbytes = WORK[part]
        print(f"  [{part}] FLOP={flop/1e9:.2f} G，必需流量={nbytes/1e6:.0f} MB，AI={flop/nbytes:.2f} FLOP/B")
        names = [k for k in VARIANTS if k.startswith(part)]
        ms = {k: T.gpu_time_ms(VARIANTS[k]) for k in names}
        for k, v in ms.items():
            print(f"    {k:<18s} {v:7.3f} ms  {nbytes / v / 1e6:6.1f} GB/s  {flop / v / 1e9:6.2f} TFLOP/s  "
                  f"({ms[names[0]] / v:.2f}x vs {names[0]})")

    print("\n== 2. ncu 采集 ==")
    rep = T.profile(__file__, "03_register_bound",
                    sections=["SpeedOfLight", "Occupancy", "LaunchStats", "WarpStateStats", "SchedulerStats",
                              "MemoryWorkloadAnalysis", "ComputeWorkloadAnalysis", "SourceCounters"],
                    metrics=T.BASE_METRICS + EXTRA)
    rows = T.merge_by_variant(T.load(rep))
    for r in rows:
        r["kernel"] = r["kernel"][:28]
    T.table(rows, [T.COL_VARIANT, ("kernel", "kernel", ""), T.COL_TIME, T.COL_REGS,
                   ("blk_lim_regs", "launch__occupancy_limit_registers", ".0f"),
                   T.COL_TOCC, T.COL_OCC,
                   ("warps/sched", "smsp__warps_active.avg.per_cycle_active", ".2f"),
                   ("eligible/sched", "smsp__warps_eligible.avg.per_cycle_active", ".2f"),
                   ("issue%", "smsp__issue_active.avg.pct_of_peak_sustained_active", ".0f")],
            title="== 3a. 寄存器 → Occupancy → 每个 scheduler 每周期就绪的 warp → Issue 利用率 ==")
    T.table(rows, [T.COL_VARIANT, T.COL_SM, ("FMA_pipe%", "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active", ".0f"),
                   T.COL_DRAM_GBS, T.COL_DRAM_PCT,
                   ("local_MB", lambda r: (r["l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum"]
                                           + r["l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum"]) * 32 / 1e6, ".0f"),
                   ("long_sb(cyc/inst)", "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio", ".2f"),
                   ("lg_thr(cyc/inst)", "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio", ".2f"),
                   ("inst(M)", lambda r: r["smsp__inst_executed.sum"] / 1e6, ".0f")],
            title="== 3b. 吞吐、spill 流量（local memory）、stall（每条已发射指令平均等待的周期）==")

    print("\n== 4. Source 页：stall 采样落在哪一行（B_per_thread vs B_prefetch）==")
    T.hot_lines(rep, r"filter128_per_thread<\(int\)0>", top=3)
    T.hot_lines(rep, r"filter128_per_thread<\(int\)4>", top=3)

    for v in ("A_per_thread", "B_per_thread"):
        print(f"\n== 5. ncu 规则引擎的建议（{v}）==")
        for s in T.rules(next(r for r in rows if r["variant"] == v))[:4]:
            print("  -", s)


if __name__ == "__main__":
    main()
