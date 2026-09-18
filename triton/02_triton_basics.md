# 第 2 章：Triton 基础

## 2.1 编程模型：program、block、grid

Triton 的核心心智模型：

- 你写的 `@triton.jit` 函数描述**一个 program 实例如何处理一块（block）数据**
- `tl.program_id(axis)` 拿到当前 program 在 grid 中的坐标，等价于 CUDA 里的 `blockIdx`
- `tl.arange(0, BLOCK_SIZE)` 生成一整块的偏移向量——**没有** CUDA C++ 里逐线程写
  `threadIdx.x` 的过程，Triton 把"一个 block 内所有线程怎么协作处理这一块数据"这件事隐藏掉了，
  你操作的是向量化的 tile，而不是单个标量
- `mask` 用来处理"数据总量不是 BLOCK_SIZE 整数倍"的边界情况——`tl.load`/`tl.store` 都接受
  `mask` 参数，超出范围的位置不会真正访存

如果你熟悉 CUDA C++，可以这样类比：

```cuda
// CUDA C++：每个线程处理一个元素，你自己算全局索引
__global__ void add_kernel(float* x, float* y, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = x[idx] + y[idx];
}
```

```python
# Triton：每个 "program" 处理一块元素，索引是向量而不是标量
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # 一整块的索引
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)
```

CUDA 版本里没有边界检查会越界；Triton 里忘了写 `mask` 同样会越界——这不是 Triton 帮你避免的坑，
而是它把"处理一块数据"这个粒度的心智负担从"这一个线程做什么"抽象成了"这一块数据整体做什么"。

## 2.2 例 1：Vector Add

完整代码见 [`code/02_vector_add.py`](code/02_vector_add.py)。关键部分：

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def triton_add(x, y):
    out = torch.empty_like(x)
    n_elements = out.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)  # 需要多少个 program 实例
    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out
```

`add_kernel[grid](...)` 这个下标语法是 Triton 的 kernel 启动方式，`grid` 相当于 CUDA
的 `<<<gridDim, blockDim>>>` 里的 `gridDim`（Triton 不需要你指定 blockDim/线程数，
`num_warps` 这类参数是可选的调优项，第 3 章会讲）。

### 实际运行

```bash
python3 code/02_vector_add.py
```

输出：

```
torch      : 2.9.1
triton     : 3.5.0
cuda avail : True
device     : NVIDIA Thor
capability : (11, 0)
------------------------------------------------------------
[PASS] vector_add: allclose=True, max_abs_err=0.000e+00
triton add : 0.0249 ms
torch  add : 0.0158 ms
```

正确性完全一致（`max_abs_err=0`，因为加法在浮点下是精确可复现的）。性能上 PyTorch 的原生
`+`（背后是高度优化的 elementwise kernel）比我们刚写的、没做任何调优的 Triton kernel 略快——
这符合预期：vector add 是内存带宽瓶颈的操作，简单实现已经接近硬件上限，写自定义 kernel
的价值不在这类极简算子上，而在于**融合多个算子**（第 3 章的 fused LayerNorm 会展示这一点）。

## 2.3 例 2：Softmax（归约 + 数值稳定性）

完整代码见 [`code/02_softmax.py`](code/02_softmax.py)。这个例子引入两个新概念：

1. **归约（reduction）**：`tl.max(row, axis=0)`、`tl.sum(numerator, axis=0)` 在一个 block
   内部做归约，等价于 CUDA 里手写的 warp/block-level reduction，但 Triton 帮你生成好了。
2. **数值稳定性**：直接算 `exp(x)` 在 x 较大时会溢出，标准做法是先减去这一行的最大值。
   这个技巧和你在任何框架（PyTorch/TensorFlow/手写 CUDA）里实现 softmax 时用的完全一样，
   Triton 不会替你做，需要显式写出来：

```python
@triton.jit
def softmax_kernel(out_ptr, in_ptr, in_row_stride, out_row_stride, n_cols,
                    BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)          # 一个 program 处理一整行
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    row_start_ptr = in_ptr + row_idx * in_row_stride
    row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float("inf"))

    row_max = tl.max(row, axis=0)
    numerator = tl.exp(row - row_max)   # 减去最大值再指数化，避免溢出
    denominator = tl.sum(numerator, axis=0)
    result = numerator / denominator

    out_row_start_ptr = out_ptr + row_idx * out_row_stride
    tl.store(out_row_start_ptr + col_offsets, result, mask=mask)
```

注意 `other=-float("inf")`：被 mask 掉的越界位置填 `-inf`，这样它们在 `tl.max` 里不会影响结果
（`tl.sum` 的分子对应位置也会是 `exp(-inf - max) = 0`，同样不影响结果）。

`BLOCK_SIZE = triton.next_power_of_2(n_cols)`：Triton 的很多归约操作要求 block size 是 2 的
幂，实际列数不是幂次时用 mask 补齐即可，这也是为什么示例故意选了 `n_cols=781` 这种"不整齐"的数字。

### 实际运行

```bash
python3 code/02_softmax.py
```

输出：

```
torch      : 2.9.1
triton     : 3.5.0
cuda avail : True
device     : NVIDIA Thor
capability : (11, 0)
------------------------------------------------------------
[PASS] softmax: allclose=True, max_abs_err=7.451e-09
triton softmax : 0.0567 ms
torch  softmax : 0.0548 ms
```

`n_cols=781`（非 2 的幂）验证了 mask 逻辑处理边界正确；误差在 `1e-8` 级别（浮点归约顺序不同
导致的正常误差）。性能与 PyTorch 内置实现基本持平——同样因为这是一个内存带宽瓶颈、
PyTorch 已经高度优化过的简单算子。

## 2.4 小结

- Triton 用"block 级向量操作"替代了 CUDA C++"线程级标量操作"，少写了索引计算和边界判断的样板代码
- `mask` 机制贯穿所有 `tl.load`/`tl.store`，是处理非整齐 shape 的标准方式
- 归约（`tl.max`/`tl.sum`）和数值稳定性技巧需要显式写出，Triton 不是"自动帮你优化数学"，
  它只是让你更方便地写出手动优化过的 GPU 代码
- 在纯 elementwise/简单归约算子上，手写 Triton kernel 和 PyTorch 内置实现性能相近——
  Triton 的真正价值在算子融合，下一章会展示

下一章：[第 3 章：Triton 进阶](03_triton_intermediate.md) —— autotune、矩阵乘法、算子融合、性能分析
