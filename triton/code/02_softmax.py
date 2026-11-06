"""Chapter 2, Example 2: row-wise softmax with numerically-stable reduction.

Demonstrates tl.max / tl.sum reductions within a block, and processing one
matrix row per program instance. Run: python3 02_softmax.py
"""
from common import device_info, check_close, bench_ms

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    out_ptr, in_ptr,
    in_row_stride, out_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # 每个 program 负责处理一整行。BLOCK_SIZE 必须 >= n_cols
    # （多出来的位置会被 mask 掉，读取时填 -inf，这样不会影响 max/sum 的结果）。
    row_idx = tl.program_id(0)  # 当前 program 处理第几行，一个 program 对应一行
    col_offsets = tl.arange(0, BLOCK_SIZE)  # 一整行的列下标向量 [0, 1, ..., BLOCK_SIZE-1]
    mask = col_offsets < n_cols  # BLOCK_SIZE 可能比实际列数大（取的是 >=n_cols 的 2 的幂），超出部分用 mask 挡住

    row_start_ptr = in_ptr + row_idx * in_row_stride  # 用 stride 而不是 n_cols 定位行首地址，兼容非连续/padding 的张量布局
    row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float("inf"))  # 读入这一整行；被 mask 掉的位置填 -inf（对 max 无影响，对 exp 后是 0）

    # 先减去行内最大值再做指数运算：数值稳定性的标准技巧，
    # 避免 exp() 直接对较大的原始值计算导致数值溢出（inf/nan），跟任何框架实现 softmax 的道理一样。
    row_max = tl.max(row, axis=0)  # 沿着这一行（axis=0，因为这里 row 是一维向量）求最大值
    numerator = tl.exp(row - row_max)  # 减去最大值后再 exp，最大值处结果为 exp(0)=1，其余 <=1，不会溢出；被 mask 的 -inf 位置 exp 后变成 0
    denominator = tl.sum(numerator, axis=0)  # 分母：这一行 exp 值的总和（被 mask 掉的 0 不影响求和）
    result = numerator / denominator  # 归一化，得到这一行的 softmax 结果，总和为 1

    out_row_start_ptr = out_ptr + row_idx * out_row_stride  # 输出这一行的起始地址（同样用 stride 定位）
    tl.store(out_row_start_ptr + col_offsets, result, mask=mask)  # 写回结果，被 mask 掉的越界位置不写


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)  # 取 >= n_cols 的最小 2 的幂，作为每行处理的向量宽度（tl.arange 通常要求 2 的幂）
    grid = (n_rows,)  # 一行一个 program，grid 大小就是行数
    softmax_kernel[grid](
        out, x,
        x.stride(0), out.stride(0),  # 传入行 stride（每行之间在内存里跨越多少元素），而不是假设行是紧密排列的
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


if __name__ == "__main__":
    device_info()

    n_rows, n_cols = 4096, 781  # n_cols 特意取奇数（非 2 的幂），用来实际触发 mask 逻辑
    x = torch.randn(n_rows, n_cols, device="cuda")

    out_triton = triton_softmax(x)
    out_torch = torch.softmax(x, dim=1)
    check_close(out_triton, out_torch, "softmax")

    t_triton = bench_ms(lambda: triton_softmax(x))
    t_torch = bench_ms(lambda: torch.softmax(x, dim=1))
    print(f"triton softmax : {t_triton:.4f} ms")
    print(f"torch  softmax : {t_torch:.4f} ms")
