"""5.4 Memory-bound：逐元素 kernel 的合并访存、向量化、融合。

    python 02_memory_bound.py            # 计时 + ncu 采集 + 对比表，输出 outputs/02_memory_bound.{log,ncu-rep}

变体（N = 64M 个 float = 256 MB）：
    torch_mul      PyTorch 的 x*2（参照）
    coalesced      y[i] = a*x[i]
    strided        按列访问（相邻线程相差 8 MB）→ 非合并
    vec4           float4 向量化
    eager_3ops     relu(a*x+b) 用 PyTorch eager 写：mul、add、relu 共 3 个 kernel
    fused          同样的计算融合成 1 个 kernel
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ncu_tools as T  # noqa: E402
from kernels import K  # noqa: E402

N = 64 * 1024 * 1024
x = torch.randn(N, device="cuda")
y = torch.empty_like(x)


def eager_3ops():
    return torch.relu(x * 2.0 + 1.0)


VARIANTS = {
    "torch_mul": lambda: torch.mul(x, 2.0, out=y),
    "coalesced": lambda: K.scale_coalesced(x, y, 2.0),
    "strided": lambda: K.scale_strided(x, y, 2.0, 32),
    "vec4": lambda: K.scale_vec4(x, y, 2.0),
    "eager_3ops": eager_3ops,
    "fused": lambda: K.fused_axpb_relu(x, y, 2.0, 1.0),
}
USEFUL_BYTES = {k: 2 * N * 4 for k in VARIANTS}   # 每个变体"必须"的流量：读一次 x、写一次 y

EXTRA = [
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",   # 全局 load 请求数（每条 warp 级 load 指令 = 1 个请求）
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",    # 这些请求涉及的 32B sector 数
    "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "lts__t_sectors_srcunit_tex.sum",                    # L1 → L2 的 sector 数
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
]


def check():
    K.scale_coalesced(x, y, 2.0); assert torch.allclose(y, x * 2)
    K.scale_strided(x, y, 2.0, 32); assert torch.allclose(y, x * 2)
    K.scale_vec4(x, y, 2.0); assert torch.allclose(y, x * 2)
    K.fused_axpb_relu(x, y, 2.0, 1.0); assert torch.allclose(y, eager_3ops())


def main():
    if T.worker_mode():
        T.run_variants(VARIANTS)
        return
    T.tee("02_memory_bound")
    check()
    print(f"N = {N/2**20:.0f}M float32 = {N*4/2**20:.0f} MB；必需流量 = 读 x + 写 y = {2*N*4/1e6:.0f} MB\n")
    print("== 1. 计时（CUDA Event，每次刷 L2，取中位数）==")
    ms = {k: T.gpu_time_ms(f) for k, f in VARIANTS.items()}
    for k, v in ms.items():
        print(f"  {k:<12s} {v:7.3f} ms   有效带宽 {USEFUL_BYTES[k] / v / 1e6:6.1f} GB/s   ({v / ms['fused']:.2f}x of fused)")

    print("\n== 2. ncu 采集 ==")
    rep = T.profile(__file__, "02_memory_bound", sections=["SpeedOfLight", "MemoryWorkloadAnalysis", "WarpStateStats"],
                    metrics=T.BASE_METRICS + EXTRA)
    rows = T.load(rep)
    for r in rows:
        r["kernel"] = r["kernel"].replace("vectorized_elementwise_kernel", "vec_elementwise")[:22]

    T.table(rows, [T.COL_VARIANT, ("kernel", "kernel", ""), T.COL_TIME, T.COL_SM, T.COL_MEM, T.COL_DRAM_RD, T.COL_DRAM_WR, T.COL_L2_WR_REQ, T.COL_DRAM_GBS,
                   T.COL_DRAM_PCT, T.COL_L2HIT],
            title="== 3a. SOL 与 DRAM 流量（DRAM 由 L2 sysmem 计数换算，见 ncu_tools.py）==")

    def spr(kind):
        return lambda r: (r[f"l1tex__t_sectors_pipe_lsu_mem_global_op_{kind}.sum"] /
                          max(r[f"l1tex__t_requests_pipe_lsu_mem_global_op_{kind}.sum"], 1))
    T.table(rows, [T.COL_VARIANT, ("kernel", "kernel", ""),
                   ("ld_requests(M)", lambda r: r["l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum"] / 1e6, ".2f"),
                   ("sectors/req(ld)", spr("ld"), ".1f"), ("sectors/req(st)", spr("st"), ".1f"),
                   ("L1→L2_MB", lambda r: r["lts__t_sectors_srcunit_tex.sum"] * 32 / 1e6, ".0f"),
                   ("warp_inst(M)", lambda r: r["smsp__inst_executed.sum"] / 1e6, ".1f"),
                   ("stall_long_sb%", "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct", ".0f")],
            title="== 3b. 访存模式：每个请求涉及几个 sector（理想值：float 为 4，float4 为 16）==")

    print("\n== 4. 融合：eager 的 3 个 kernel 加起来 vs 1 个融合 kernel ==")
    for v in ("eager_3ops", "fused"):
        rs = [r for r in rows if r["variant"] == v]
        t = sum(r["gpu__time_duration.sum"] for r in rs) / 1e3
        mb = sum(T.sysmem_bytes(r) for r in rs) / 1e6
        print(f"  {v:<11s} {len(rs)} 个 kernel  ncu 时间合计 {t:8.1f} us  DRAM 流量合计 {mb:7.1f} MB  "
              f"各 kernel 的 DRAM%(实测峰值) = {[round(100 * T.sysmem_gbs(r) / T.PEAK_BW_GBS) for r in rs]}")

    print("\n== 5. ncu 规则引擎的建议（strided）==")
    for s in T.rules(next(r for r in rows if r["variant"] == "strided"))[:3]:
        print("  -", s)


if __name__ == "__main__":
    main()
