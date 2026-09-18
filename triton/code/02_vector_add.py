"""Chapter 2, Example 1: vector add — the "hello world" of Triton kernels.

Run: python3 02_vector_add.py
"""
from common import device_info, check_close, bench_ms

import torch
import triton
import triton.language as tl

# 一句话：每个 program 负责一段连续的 BLOCK_SIZE 元素，用向量化的 load-compute-store 完成加法，mask 处理尾部不整除的情况。
@triton.jit
def add_kernel(
    x_ptr, y_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # 每个"program"（Triton 对一个 kernel 实例的称呼）负责处理一段
    # 连续的 BLOCK_SIZE 个元素。这里没有像 CUDA C++ 那样手写线程索引计算——
    # tl.arange 直接一次性给出整个 block 的偏移向量。
    pid = tl.program_id(axis=0)  # 当前 program 的编号，范围 0..grid[0]-1，类似 CUDA 里的 blockIdx.x。
    block_start = pid * BLOCK_SIZE  # 当前 program 负责的第一个全局下标
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # 当前 program 要处理的 BLOCK_SIZE 个下标（向量），生成一个长度为 BLOCK_SIZE 的向量 [block_start, block_start+1, ..., block_start+BLOCK_SIZE-1]，即这个 program 要处理的所有下标。这是 Triton 的特色：不用像 CUDA 那样手写线程索引，tl.arange 直接给出整块的偏移向量，后续操作都是向量化的。
    mask = offsets < n_elements  # 防止最后一个 block 越界（可能不是整块的），因为 n_elements 不一定能被 BLOCK_SIZE 整除，最后一个 block 可能会越界，所以生成一个布尔掩码，标记哪些下标是"有效"的。

    x = tl.load(x_ptr + offsets, mask=mask)  # 按 offsets 读取 x，被 mask 掉的位置读到 0。按偏移量把数据从全局内存读到寄存器/向量中；mask=False 的位置不会真正读取越界内存（避免非法内存访问）。
    y = tl.load(y_ptr + offsets, mask=mask)  # 按 offsets 读取 y
    tl.store(out_ptr + offsets, x + y, mask=mask)  # 写回结果，被 mask 掉的位置不写


def triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)  # 先分配一个和 x 形状/dtype/device 相同的输出张量。未初始化；kernel 会写满每个元素
    n_elements = out.numel() #元素总数
    BLOCK_SIZE = 1024  # 每个 program 处理的元素数；一般取 2 的幂，可调
    # grid = 启动的 program 实例数量，一个 block 对应一个 program；
    # cdiv 向上取整，保证最后不满一个 block 时也能被覆盖（对应上面的 mask）
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),) # n_elements除以BLOCK_SIZE，向上取整，保证即使 n_elements 不能被 BLOCK_SIZE 整除，也有足够多的 block 覆盖所有元素
    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE) # 这是 Triton 的启动语法，[grid] 指定网格大小，然后像调用函数一样传参数。BLOCK_SIZE 用关键字传入，因为它是 tl.constexpr，Triton 会针对这个具体值编译特化版本的 kernel。
    return out


if __name__ == "__main__":
    device_info()

    n = 2**20  # 约 100 万个元素
    x = torch.rand(n, device="cuda")
    y = torch.rand(n, device="cuda")

    out_triton = triton_add(x, y)
    out_torch = x + y
    check_close(out_triton, out_torch, "vector_add")

    t_triton = bench_ms(lambda: triton_add(x, y))
    t_torch = bench_ms(lambda: x + y)
    print(f"triton add : {t_triton:.4f} ms")
    print(f"torch  add : {t_torch:.4f} ms")
