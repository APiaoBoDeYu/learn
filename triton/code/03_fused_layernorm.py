"""Chapter 3, Example 2: fused LayerNorm — one kernel instead of several.

Naive LayerNorm in PyTorch eager triggers separate kernels for mean, var,
normalize, scale+shift — each a full read+write round-trip to HBM. Fusing
them into a single Triton kernel is the same idea as TensorRT's layer/kernel
fusion, but written explicitly by us instead of decided by a builder.
Run: python3 03_fused_layernorm.py
"""
from common import device_info, check_close, bench_ms

import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    row_stride,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # 同 softmax kernel 的思路：一个 program 处理一整行。
    row_idx = tl.program_id(0)  # 当前 program 负责第几行
    col_offsets = tl.arange(0, BLOCK_SIZE)  # 一整行的列下标向量
    mask = col_offsets < n_cols  # BLOCK_SIZE 是 >=n_cols 的 2 的幂，超出实际列数的部分要挡住

    row_ptr = x_ptr + row_idx * row_stride  # 用 stride 定位这一行的起始地址（不假设行是紧密排列的）
    x = tl.load(row_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)  # 读入整行，越界处填 0；统一转 fp32 做归约，避免低精度累加误差

    mean = tl.sum(x, axis=0) / n_cols  # 行均值：注意除的是真实列数 n_cols，不是 BLOCK_SIZE（padding 的 0 不能算进分母）
    diff = tl.where(mask, x - mean, 0.0)  # x - mean；显式把 padding 位置的差值清零，防止它们污染下面的方差求和
    var = tl.sum(diff * diff, axis=0) / n_cols  # 行方差：E[(x-mean)^2]
    inv_std = 1.0 / tl.sqrt(var + eps)  # 标准差的倒数；加 eps 防止 var=0 时除零
    x_norm = diff * inv_std  # 归一化：(x - mean) / std，得到均值0方差1的分布

    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0).to(tl.float32)  # LayerNorm 的可学习缩放参数 gamma（越界位置填1，无实际意义，反正后面会被 mask 挡住不写出）
    bias = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)  # LayerNorm 的可学习偏移参数 beta
    y = x_norm * weight + bias  # 逐元素做仿射变换：LayerNorm 公式的最后一步 y = x_norm * gamma + beta

    out_row_ptr = out_ptr + row_idx * row_stride  # 输出这一行的起始地址
    tl.store(out_row_ptr + col_offsets, y, mask=mask)  # 写回结果，被 mask 掉的越界位置不写


def triton_layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                      eps: float = 1e-5) -> torch.Tensor:
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)  # 取 >=n_cols 的最小 2 的幂，保证一个 program 能一次性装下整行
    grid = (n_rows,)  # 一行一个 program
    layernorm_kernel[grid](
        x, weight, bias, out,
        x.stride(0), n_cols, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


if __name__ == "__main__":
    device_info()

    n_rows, n_cols = 4096, 1024
    x = torch.randn(n_rows, n_cols, device="cuda")
    weight = torch.randn(n_cols, device="cuda")
    bias = torch.randn(n_cols, device="cuda")

    out_triton = triton_layernorm(x, weight, bias)
    out_torch = torch.nn.functional.layer_norm(x, (n_cols,), weight, bias)
    check_close(out_triton, out_torch, "fused_layernorm")

    def naive_layernorm():
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + 1e-5)
        return x_norm * weight + bias

    t_fused = bench_ms(lambda: triton_layernorm(x, weight, bias))
    t_naive = bench_ms(naive_layernorm)
    t_torch = bench_ms(lambda: torch.nn.functional.layer_norm(x, (n_cols,), weight, bias))
    print(f"triton fused layernorm : {t_fused:.4f} ms")
    print(f"naive multi-kernel     : {t_naive:.4f} ms")
    print(f"torch  F.layer_norm    : {t_torch:.4f} ms")
