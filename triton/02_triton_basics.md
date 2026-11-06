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

### `out_ptr` 是 global memory，不是 shared memory

`x_ptr`/`y_ptr`/`out_ptr` 都是**global memory**上的指针，对应 CUDA 里 `add_kernel(float* x, ...)`
参数里的 `x`/`y`/`out`——都是 host 端用 `cudaMalloc`（或者 PyTorch tensor 底层）分配、GPU 上所有
block 都能访问的显存。`tl.load`/`tl.store` 默认操作的就是 global memory，`out[idx] = ...` 和
`tl.store(out_ptr + offsets, ...)` 是同一层内存。

这个 kernel 完全没用到 shared memory（CUDA 里的 `__shared__`）——vector add 每个元素只读一次、
写一次，没有"block 内线程互相复用数据"的场景，自然不需要 shared memory；2.3 节 softmax 的归约
（`tl.max`/`tl.sum`）才是需要 block 内数据共享的场景，Triton 编译器会在背后自动用 shared memory
实现这个归约，但这一层对你完全透明，你的代码里看不到任何 `__shared__` 或者显存层级的字样。

### 把 Triton kernel 逐行翻译成 CUDA

如果按 Triton 这段代码的粒度（一个 program = 一个 block，`BLOCK_SIZE` 个元素）逐行翻译成
CUDA C++，并且让 `blockDim.x == BLOCK_SIZE`（即一个 block 里的线程数正好等于 Triton 的
`BLOCK_SIZE`，这也是 `add_kernel[grid](...)` 启动时 Triton 编译器背后实际做的映射），
会得到和最开始那段 CUDA 代码完全一样的东西——这不是巧合，说明"一个 program 处理一块数据"
展开到硬件层，本质就是"一个 block 里每个线程各处理块内一个元素"：

```cuda
// 对 Triton add_kernel 逐行翻译，BLOCK_SIZE == blockDim.x
__global__ void add_kernel(float* x_ptr, float* y_ptr, float* out_ptr, int n_elements) {
    int pid = blockIdx.x;                              // pid = tl.program_id(0)
    int block_start = pid * blockDim.x;                 // pid * BLOCK_SIZE

    // tl.arange(0, BLOCK_SIZE) 在 CUDA 里没有对应的"一整块偏移向量"，
    // 因为向量的每一个分量本来就是由不同线程各自持有一个标量 threadIdx.x
    int offset = block_start + threadIdx.x;             // offsets（这里是标量，对应向量里"我这个线程负责的那一个分量"）
    bool mask = offset < n_elements;                    // mask（同样是标量，Triton 里是整块向量的逐元素判断）

    if (mask) {                                         // tl.load/tl.store 的 mask 参数在 CUDA 里就是 if 判断
        float x = x_ptr[offset];                         // tl.load(x_ptr + offsets, mask=mask)
        float y = y_ptr[offset];                         // tl.load(y_ptr + offsets, mask=mask)
        out_ptr[offset] = x + y;                          // tl.store(out_ptr + offsets, x + y, mask=mask)
    }
}
```

关键区别就在于：Triton 里 `offsets`/`mask`/`x`/`y` 都是长度为 `BLOCK_SIZE` 的**向量**，一行代码
处理整个 block 的数据；翻译成 CUDA 后，同样的变量在每个线程里都只是一个**标量**，"整个 block
处理这块数据"这件事是靠 `blockDim.x` 个线程各自执行同一份代码、隐式地拼成的——Triton 的 `mask`
参数其实就是把 CUDA 里那个 `if (idx < n)` 从"每个线程自己判断一次"变成了"整块向量一次性打包判断"。

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

### `grid` 和 `block` 的区别：一个是数量，一个是资源单位

容易搞混的一点：`grid` 看起来就是"block 的数量"，这个直觉没错，但要分清楚 `grid` 和 `block`
在这件事上的角色完全不同：

- **`grid`** 只回答"要启动多少个 block 实例"——本身**不分配任何硬件资源**，只是一个调度用的
  坐标系/计数（可以是 1D/2D/3D，比如第 3 章矩阵乘法会用 `(num_pid_m, num_pid_n)` 这种二维 grid）。
- **`block`**（连同它的大小，也就是 `BLOCK_SIZE` 对应的线程数）才是**真正分配资源的单位**——
  寄存器、shared memory、线程都是按 block 分配的，GPU 调度器也是以 block 为粒度把任务派发到
  某个 SM 上执行。

对应到 CUDA 就是：

```cuda
int num_blocks = (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;  // 对应 triton.cdiv，也就是 grid
add_kernel<<<num_blocks, BLOCK_SIZE>>>(x, y, out, n_elements);
//           ^^^^^^^^^^  ^^^^^^^^^^
//           这就是 grid   这就是 blockDim，真正决定每个 block 用多少线程/占多少资源
```

`triton.cdiv(n_elements, BLOCK_SIZE)` 算的正是"数据总量除以每块大小，向上取整"——也就是"需要几个
block 才能覆盖所有数据"，这也是为什么在 vector add 这个例子里 `grid` 看起来就等于 block 数量：
因为这里 grid 唯一的作用就是表达这一件事。一句话区分：**grid 回答"跑几个 block"，block（连同
`BLOCK_SIZE`）才回答"每个 block 用多少线程、占多少资源"**——grid 本身没有"资源"概念，纯粹是迭代
次数/坐标系。

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

`tl.max(row, axis=0)` / `tl.sum(numerator, axis=0)` 这两行看起来只是函数调用，但它们背后替你做的事情
在 CUDA C++ 里要手写共享内存 + 树形归约 + 显式同步。同样处理"一行 softmax"，CUDA 版本大致是这样：

```cuda
// CUDA C++：一个 block 处理一行，block 内多个线程协作做 reduction
__global__ void softmax_kernel(float* out, const float* in, int n_cols) {
    int row = blockIdx.x;                        // 等价于 Triton 的 tl.program_id(0)
    extern __shared__ float shared[];             // 手动申请归约用的共享内存
    const float* row_ptr = in + row * n_cols;
    float* out_ptr = out + row * n_cols;

    // 1) 每个线程先在自己负责的列上求局部最大值（列数可能比线程数多，需要跨步循环）
    float local_max = -INFINITY;
    for (int col = threadIdx.x; col < n_cols; col += blockDim.x)
        local_max = fmaxf(local_max, row_ptr[col]);
    shared[threadIdx.x] = local_max;
    __syncthreads();                              // 等所有线程写完局部结果

    // 2) block 内树形归约：手动写 log2(blockDim.x) 轮，才能得到整行的最大值
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
        __syncthreads();                          // 每一轮归约后都要同步
    }
    float row_max = shared[0];
    __syncthreads();                              // 所有线程读完 shared[0] 才能进入下一步复用它

    // 3) 同样的模式再做一次，算 exp 之和
    float local_sum = 0.0f;
    for (int col = threadIdx.x; col < n_cols; col += blockDim.x)
        local_sum += expf(row_ptr[col] - row_max);
    shared[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    float row_sum = shared[0];
    __syncthreads();

    // 4) 每个线程写回自己负责的那部分结果
    for (int col = threadIdx.x; col < n_cols; col += blockDim.x)
        out_ptr[col] = expf(row_ptr[col] - row_max) / row_sum;
}
```

对比一下就能看清 Triton 隐藏了什么：
- CUDA 里 `n_cols` 可能大于 `blockDim.x`（线程数），所以每个线程要用 `for (... += blockDim.x)`
  的跨步循环处理多个列；Triton 里 `BLOCK_SIZE >= n_cols`（配合 `mask`），一个 program 用一条向量
  指令就覆盖整行，不需要这层循环。
- `tl.max`/`tl.sum` 对应的是上面那一整块「共享内存 + 树形归约 + 每轮 `__syncthreads()`」——
  这是最容易写错的地方（少一个 `__syncthreads()` 就会读到还没写完的中间结果），Triton 编译器
  在背后帮你生成等价逻辑，你不需要关心线程数是不是 2 的幂、`shared` 数组多大、同步几次。
- `row_max`/`row_sum` 在 CUDA 里是"归约完成后存在 `shared[0]` 里，每个线程各自读一次"；
  在 Triton 里 `tl.max`/`tl.sum` 的返回值本身就是一个标量，直接参与后续向量运算
  （`numerator / denominator` 是整块向量除以一个标量的广播运算）。

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
