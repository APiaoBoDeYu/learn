"""5.7 Shared memory bank conflict：矩阵转置（8192 x 8192 float32，读 256 MB + 写 256 MB）。

    python 05_bank_conflict.py

    copy           参照：纯拷贝（读写都合并），是转置能达到的上限
    naive          直接转置：读合并、写不合并
    smem           经 shared memory 中转：读写都合并，但 tile[32][32] 按列读 → 32-way bank conflict
    smem_pad       tile[32][33]：每行错开一个 bank
    smem_swizzle   tile[32][32] + XOR swizzle：不浪费 shared memory
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ncu_tools as T  # noqa: E402
from kernels import K  # noqa: E402

R = Cc = 8192
inp = torch.randn(R, Cc, device="cuda")
out = torch.empty(Cc, R, device="cuda")
out_copy = torch.empty(R, Cc, device="cuda")

VARIANTS = {
    "copy": lambda: K.transpose(4, inp, out_copy),
    "naive": lambda: K.transpose(0, inp, out),
    "smem": lambda: K.transpose(1, inp, out),
    "smem_pad": lambda: K.transpose(2, inp, out),
    "smem_swizzle": lambda: K.transpose(3, inp, out),
}

EXTRA = [
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",      # shared load 实际花费的 wavefront（≈ 周期）数
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
    "smsp__inst_executed_op_shared_ld.sum",                     # shared load 指令数（warp 级）
    "smsp__inst_executed_op_shared_st.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared.avg.pct_of_peak_sustained_elapsed",
]


def check():
    ref = inp.t().contiguous()
    for v in range(4):
        out.zero_(); K.transpose(v, inp, out)
        assert torch.equal(out, ref), v
    K.transpose(4, inp, out_copy); assert torch.equal(out_copy, inp)


def main():
    if T.worker_mode():
        T.run_variants(VARIANTS)
        return
    T.tee("05_bank_conflict")
    check()
    nbytes = 2 * R * Cc * 4
    print(f"== 1. 计时（{R}x{Cc} float32，必需流量 {nbytes/1e6:.0f} MB，CUDA Event + 刷 L2）==")
    ms = {k: T.gpu_time_ms(f) for k, f in VARIANTS.items()}
    ms["torch .t().contiguous()"] = T.gpu_time_ms(lambda: inp.t().contiguous())
    for k, v in ms.items():
        print(f"  {k:<24s} {v:7.3f} ms  {nbytes / v / 1e6:6.1f} GB/s  ({ms['naive'] / v:.2f}x vs naive)")

    print("\n== 2. ncu 采集 ==")
    rep = T.profile(__file__, "05_bank_conflict",
                    sections=["SpeedOfLight", "MemoryWorkloadAnalysis", "MemoryWorkloadAnalysis_Tables",
                              "WarpStateStats", "SourceCounters"],
                    metrics=T.BASE_METRICS + EXTRA)
    rows = T.load(rep)
    for r in rows:
        r["kernel"] = r["kernel"][:22]

    def ratio(a, b):
        return lambda r: r[a] / r[b] if r.get(b) else None
    T.table(rows, [T.COL_VARIANT, T.COL_TIME, T.COL_DRAM_RD, T.COL_DRAM_WR, T.COL_L2_WR_REQ, T.COL_DRAM_GBS, T.COL_DRAM_PCT,
                   ("ld_sectors/req", ratio("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
                                            "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum"), ".1f"),
                   ("st_sectors/req", ratio("l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
                                            "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum"), ".1f")],
            title="== 3a. 全局内存：合并程度（理想值 4）与带宽 ==")
    T.table(rows, [T.COL_VARIANT,
                   ("smem_ld_inst(M)", lambda r: r["smsp__inst_executed_op_shared_ld.sum"] / 1e6, ".2f"),
                   ("ld_wavefronts(M)", lambda r: r["l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum"] / 1e6, ".2f"),
                   ("wavefronts/ld", ratio("l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum", "smsp__inst_executed_op_shared_ld.sum"), ".1f"),
                   ("ld_conflicts(M)", lambda r: r["l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"] / 1e6, ".2f"),
                   ("st_conflicts(M)", lambda r: r["l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum"] / 1e6, ".2f"),
                   ("smem_pipe%", "l1tex__data_pipe_lsu_wavefronts_mem_shared.avg.pct_of_peak_sustained_elapsed", ".0f"),
                   ("mio_thr", "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio", ".1f"),
                   ("short_sb", "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio", ".1f"),
                   ("barrier", "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio", ".1f"),
                   ("long_sb", "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio", ".1f")],
            title="== 3b. shared memory：每条 load 指令花了几个 wavefront（无冲突 = 1，32-way 冲突 = 32）；stall 单位：周期/指令 ==")

    print("\n== 4. Source 页：bank conflict 发生在哪一行（kernel 用 -lineinfo 编译）==")
    for kname in ("transpose_smem", "transpose_smem_pad"):
        T.hot_lines(rep, f"^(void )?{kname}\\(", metric="L1 Wavefronts Shared Excessive", top=1,
                    extra_cols=("L1 Wavefronts Shared", "L1 Wavefronts Shared Ideal"))

    print("\n== 5. ncu 规则引擎的建议（smem）==")
    for s in T.rules(next(r for r in rows if r["variant"] == "smem"), min_speedup=1)[:4]:
        print("  -", s)


if __name__ == "__main__":
    main()
