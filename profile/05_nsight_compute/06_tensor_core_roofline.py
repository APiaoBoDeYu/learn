"""5.8 Tensor Core 与 Roofline：确认 GEMM 走了 Tensor Core，并把本章所有 kernel 放到第 00 章的实测 Roofline 上。

    python 06_tensor_core_roofline.py

GEMM（M=N=K=4096，torch.matmul → cuBLAS）：
    fp32          关闭 TF32：只能用 CUDA core 的 FFMA
    tf32          allow_tf32=True：fp32 输入，Tensor Core 以 TF32 精度计算
    fp16 / bf16   半精度 Tensor Core
    fp16_K4095    K = 4095（不是 8 的倍数）：看 cuBLAS 选了什么 kernel、还能不能用 Tensor Core

然后读取 02~06 的报告，用 ncu 统计的 FLOP 和本章估算的 DRAM 字节画 Roofline → outputs/roofline_ncu.png
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ncu_tools as T  # noqa: E402

M = N = K = 4096
a32, b32 = torch.randn(M, K, device="cuda"), torch.randn(K, N, device="cuda")
a16, b16 = a32.half(), b32.half()
abf, bbf = a32.bfloat16(), b32.bfloat16()
a_odd, b_odd = torch.randn(M, K - 1, device="cuda").half(), torch.randn(K - 1, N, device="cuda").half()


def mm_fp32():
    torch.backends.cuda.matmul.allow_tf32 = False
    return a32 @ b32


def mm_tf32():
    torch.backends.cuda.matmul.allow_tf32 = True
    r = a32 @ b32
    torch.backends.cuda.matmul.allow_tf32 = False
    return r


VARIANTS = {
    "fp32": mm_fp32,
    "tf32": mm_tf32,
    "fp16": lambda: a16 @ b16,
    "bf16": lambda: abf @ bbf,
    "fp16_K4095": lambda: a_odd @ b_odd,
}
FLOP = {k: 2 * M * N * K for k in VARIANTS}
FLOP["fp16_K4095"] = 2 * M * N * (K - 1)

TC_OPS = [f"sm__ops_path_tensor_op_{p}_src_{s}.sum" for p in ("hmma", "utchmma")
          for s in ("tf32_dst_fp32", "fp16_dst_fp32", "fp16_dst_fp16", "bf16_dst_fp32")]
FP32_OPS = ["smsp__sass_thread_inst_executed_op_ffma_pred_on.sum", "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
            "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum"]
EXTRA = TC_OPS + FP32_OPS + [
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_tensor.sum", "sm__inst_executed_pipe_tc.sum",
    "launch__grid_size", "launch__block_size", "launch__shared_mem_per_block_dynamic",
]


def flop_fp32(r):
    g = lambda k: r.get(k) or 0  # noqa: E731
    return 2 * g(FP32_OPS[0]) + g(FP32_OPS[1]) + g(FP32_OPS[2])


def flop_tc(r):
    # ops_path_tensor 统计的是"运算次数"，下面用 2*M*N*K 校验它是按 FLOP（乘、加各算 1 次）计数的
    return sum(r.get(k) or 0 for k in TC_OPS)


def roofline_points():
    """从 02~06 的报告中取出每个 kernel 的 (AI, FLOP/s)。"""
    pts = []
    for name in ("02_memory_bound", "03_register_bound", "04_control_flow", "05_bank_conflict", "06_tensor_core"):
        rep = T.OUT / f"{name}.ncu-rep"
        if not rep.exists():
            continue
        need = T.BASE_METRICS + FP32_OPS + TC_OPS
        for r in T.merge_by_variant(T.load(rep, metrics=need)):
            f = flop_fp32(r) + flop_tc(r)
            b = T.sysmem_bytes(r)
            t = r["gpu__time_duration.sum"] * 1e-9
            if f <= 0 or b <= 0:
                continue
            pts.append({"chapter": name[:2], "variant": r["variant"], "ai": f / b, "flops": f / t,
                        "tc": flop_tc(r) > flop_fp32(r)})
    return pts


def main():
    if T.worker_mode():
        T.run_variants(VARIANTS, warmup=3)
        return
    T.tee("06_tensor_core")
    print("== 1. 计时（CUDA Event，中位数，不刷 L2）==")
    ms = {k: T.gpu_time_ms(f, flush=False) for k, f in VARIANTS.items()}
    for k, v in ms.items():
        print(f"  {k:<11s} {v:7.3f} ms  {FLOP[k] / v / 1e9:8.1f} TFLOP/s")
    torch.backends.cuda.matmul.allow_tf32 = False

    print("\n== 2. ncu 采集 ==")
    rep = T.profile(__file__, "06_tensor_core", sections=["SpeedOfLight", "ComputeWorkloadAnalysis", "InstructionStats",
                                                          "LaunchStats", "Occupancy"],
                    metrics=T.BASE_METRICS + EXTRA)
    rows = T.merge_by_variant(T.load(rep))
    for r in rows:
        r["kernel"] = r["kernel"][:34]
    T.table(rows, [T.COL_VARIANT, ("kernel", "kernel", ""), T.COL_TIME,
                   ("TFLOP/s", lambda r: FLOP[r["variant"]] / r["gpu__time_duration.sum"] / 1e3, ".1f"),
                   T.COL_SM, ("tensor_pipe%", "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active", ".1f"),
                   ("tc_pipe%", "sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_active", ".1f"),
                   ("fma_pipe%", "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active", ".1f"),
                   ("TC_ops/2MNK", lambda r: flop_tc(r) / FLOP[r["variant"]], ".3f"),
                   ("FP32_ops/2MNK", lambda r: flop_fp32(r) / FLOP[r["variant"]], ".3f"),
                   T.COL_REGS, ("smem_dyn_KB", lambda r: (r.get("launch__shared_mem_per_block_dynamic") or 0) / 1024, ".0f")],
            title="== 3. 用了哪条流水线：Tensor Core 运算数 / 理论 FLOP（=1 表示全部走 Tensor Core）==")
    for r in rows:
        tc = {k.split("src_")[1].replace(".sum", ""): v for k, v in r.items() if k in TC_OPS and v}
        print(f"  {r['variant']:<11s} 非零的 Tensor Core 运算类型: {tc or '无'}")

    print("\n== 4. Roofline（读取 02~06 的报告）==")
    pts = roofline_points()
    for p in pts:
        print(f"  [{p['chapter']}] {p['variant']:<18s} AI={p['ai']:8.2f} FLOP/B   {p['flops'] / 1e9:9.1f} GFLOP/s   {'TC' if p['tc'] else ''}")
    import json
    (T.OUT / "roofline_points.json").write_text(json.dumps(pts, indent=1))
    print(f"  已保存 {T.OUT / 'roofline_points.json'}（画图：python 06_plot_roofline.py）")


if __name__ == "__main__":
    main()
