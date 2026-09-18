"""Chapter 3, Example 1: tiled matmul with @triton.autotune.

Demonstrates tl.dot (tensor-core matmul primitive) and autotune, which is
conceptually similar to TensorRT's kernel/tactic auto-selection but operates
at the level of *source-level tile parameters* Triton will compile and
benchmark for you. Run: python3 03_matmul_autotune.py
"""
from common import device_info, check_close, bench_ms

import torch
import triton
import triton.language as tl


def get_autotune_configs():
    # BLOCK_M/BLOCK_N：输出矩阵 C 每个 tile 的行数/列数（每个 program 一次算出多大的一块 C）。
    # BLOCK_K：归约维度 K 每次切片的长度（K 很大，不能一次读完，要分段累加，见 kernel 里的 for 循环）。
    # 这里穷举 2x2x2x2=16 种 (BLOCK_M, BLOCK_N, BLOCK_K, num_warps) 组合，作为 autotune 的候选空间；
    # 最优组合和 GPU 型号、矩阵尺寸、数据类型都有关，没有公式能直接算出来，只能靠实测挑选。
    configs = []
    for BM in [64, 128]:
        for BN in [64, 128]:
            for BK in [32, 64]:
                for num_warps in [4, 8]:
                    configs.append(
                        triton.Config(
                            {"BLOCK_M": BM, "BLOCK_N": BN, "BLOCK_K": BK},
                            num_warps=num_warps,
                        )
                    )
    return configs


# @triton.autotune：第一次遇到某个 (M, N, K) 尺寸时，把 configs 里每一种配置都实际编译并跑一遍
# benchmark，选出最快的缓存下来；key=["M","N","K"] 表示只要这三个 shape 不变就复用缓存结果，
# 不用重新搜索。这与 TensorRT builder 阶段给每层尝试多种 tactic、实测选最优的思路一致，
# 区别是 TensorRT 选的是预编译好的算子实现，这里选的是要重新编译的源码级 tile 参数。
@triton.autotune(configs=get_autotune_configs(), key=["M", "N", "K"])
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
    b_ptrs = b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a_mask = (rm[:, None] < M) & (rk[None, :] + k < K)
        b_mask = (rk[:, None] + k < K) & (rn[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc = tl.dot(a, b, acc)  # tensor-core matmul, accumulate in fp32
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


if __name__ == "__main__":
    device_info()

    M, K, N = 1024, 1024, 1024
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)

    out_triton = triton_matmul(a, b)
    out_torch = (a.float() @ b.float())
    check_close(out_triton, out_torch, "matmul", atol=1e-1, rtol=5e-2)

    best_config = matmul_kernel.best_config
    print(f"best autotune config: {best_config}")

    t_triton = bench_ms(lambda: triton_matmul(a, b))
    t_torch = bench_ms(lambda: a @ b)
    flops = 2 * M * N * K  # multiply-add counted as 2 flops
    # t is in milliseconds: GFLOP/s = flops / (t_ms * 1e-3) / 1e9 = flops / t_ms / 1e6
    print(f"triton matmul : {t_triton:.4f} ms  ({flops / t_triton / 1e6:.1f} GFLOP/s)")
    print(f"torch  matmul : {t_torch:.4f} ms  ({flops / t_torch / 1e6:.1f} GFLOP/s)  (cuBLAS via tensor cores)")
