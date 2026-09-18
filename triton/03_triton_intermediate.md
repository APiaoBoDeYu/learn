# 第 3 章：Triton 进阶

## 3.1 Autotune：类比 TensorRT 的 tactic 自动选型

TensorRT 在 build engine 时会为每一层尝试多种"tactic"（不同的 kernel 实现/参数组合），
挑一个在目标硬件上最快的编进 engine——这个过程是黑盒的，你看不到候选集合。

Triton 的 `@triton.autotune` 做的是同一件事，但**发生在源码层面、由你定义候选空间**：
你给出一组 `triton.Config`（tile 大小、warp 数等参数组合），Triton 在**第一次调用**时
把每个 config 都跑一遍 benchmark，缓存最快的那个，后续调用直接用缓存结果。

```python
def get_autotune_configs():
    configs = []
    for BM in [64, 128]:
        for BN in [64, 128]:
            for BK in [32, 64]:
                for num_warps in [4, 8]:
                    configs.append(triton.Config(
                        {"BLOCK_M": BM, "BLOCK_N": BN, "BLOCK_K": BK},
                        num_warps=num_warps,
                    ))
    return configs

@triton.autotune(configs=get_autotune_configs(), key=["M", "N", "K"])
@triton.jit
def matmul_kernel(...):
    ...
```

`BLOCK_M`/`BLOCK_N` 是输出矩阵 C 每个 tile 的行数/列数（每个 program 一次算出多大的一块 C）；
`BLOCK_K` 是归约维度 K 每次切片的长度（K 通常很大，不能一次读进片上内存，要分段循环累加，
对应 3.2 节 kernel 里的 `for k in range(0, K, BLOCK_K)`）。`num_warps` 决定每个 program 用
多少个 warp 执行。这四个参数的最优组合和 GPU 型号、矩阵尺寸、数据类型都相关，没有公式能
直接算出来，只能靠 autotune 实测挑选——这正是它和 TensorRT tactic 选型类比的地方。

`key=["M", "N", "K"]` 表示：只要这三个 shape 参数不变，就复用已经选好的最优 config；
shape 变了会重新触发一次 autotune 搜索。这与 torch.compile 的 "shape 变化触发 recompile"
是同一种思路（第 5 章会讲 torch.compile 那一侧的 guard 机制）。

## 3.2 例 1：Tiled Matmul with `tl.dot`

完整代码见 [`code/03_matmul_autotune.py`](code/03_matmul_autotune.py)。核心是分块累加：

```python
@triton.autotune(configs=get_autotune_configs(), key=["M", "N", "K"])
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                   stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
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
        acc = tl.dot(a, b, acc)   # 调用张量核心（tensor core）做矩阵乘累加
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)
```

`tl.dot` 是关键：它把这一块的乘累加映射到 GPU 的张量核心（Tensor Core）指令上，
这是"用 CUDA C++ 手写 tensor core 矩阵乘"这件通常很繁琐的事情，Triton 一行搞定。
两层 grid 维度（`pid_m`, `pid_n`）对应输出矩阵的行块/列块，K 维在 kernel 内部循环累加。

### 实际运行

```bash
python3 code/03_matmul_autotune.py
```

输出：

```
torch      : 2.9.1
triton     : 3.5.0
cuda avail : True
device     : NVIDIA Thor
capability : (11, 0)
------------------------------------------------------------
[PASS] matmul: allclose=True, max_abs_err=2.747e-04
best autotune config: BLOCK_M: 64, BLOCK_N: 128, BLOCK_K: 64, num_warps: 4, num_ctas: 1, num_stages: 3, maxnreg: None
triton matmul : 0.0914 ms  (23489.3 GFLOP/s)
torch  matmul : 0.0329 ms  (65281.0 GFLOP/s)  (cuBLAS via tensor cores)
```

误差在 `1e-4` 级别是 fp16 输入、fp32 累加的正常精度损失。性能上，我们这个教学版 matmul
（1024×1024×1024）达到约 23.5 TFLOP/s，是 cuBLAS（背后是 NVIDIA 多年调优、支持任意
shape 的生产级 GEMM 库，约 65 TFLOP/s）的三分之一左右——**这是预期结果**：写一个能超过
cuBLAS 的 matmul 需要处理远比这个教学例子复杂的优化（swizzling、多级流水线、针对特定
shape 的特化等）。这里的重点是理解 `tl.dot` + autotune 的编程模型，不是造轮子超越 cuBLAS。
实践中，Triton matmul 真正大放异彩的场景是**融合了额外计算的变种**（比如 flash attention
里的 fused matmul+softmax+matmul），这种融合 cuBLAS 做不到。

## 3.3 例 2：Fused LayerNorm —— kernel 融合的价值

完整代码见 [`code/03_fused_layernorm.py`](code/03_fused_layernorm.py)。这是本章最能体现
"为什么要手写 Triton kernel"的例子。

Naive LayerNorm（用 PyTorch 逐步写）会产生多个独立 kernel：算 mean、算 var、减均值除标准差、
乘 weight 加 bias——每一步都是一次完整的"从显存读、算、写回显存"往返。融合成一个 kernel 后，
数据只读一次、写一次：

```python
@triton.jit
def layernorm_kernel(x_ptr, weight_ptr, bias_ptr, out_ptr, row_stride, n_cols, eps,
                      BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    row_ptr = x_ptr + row_idx * row_stride
    x = tl.load(row_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / n_cols
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / n_cols
    inv_std = 1.0 / tl.sqrt(var + eps)
    x_norm = diff * inv_std

    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0).to(tl.float32)
    bias = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    y = x_norm * weight + bias

    out_row_ptr = out_ptr + row_idx * row_stride
    tl.store(out_row_ptr + col_offsets, y, mask=mask)
```

这和 TensorRT 的**层融合（layer fusion）**是同一个优化思路——减少中间结果的显存往返。
区别是 TensorRT 的融合规则是内置的、你看不到具体融合了什么；这里融合的边界是你自己决定
并手写的。

### 为什么多 kernel 之间必须经过 global memory

一个自然的疑问：naive 版本里 mean、var 这些中间结果，能不能不写回显存，直接"传指针"
给下一个 kernel？答案是——**指针可以传，但指针指向的存储必须是 kernel 退出后依然存在
的存储，也就是 global memory**，不能是 register 或 shared memory。原因：

- **Register 是 per-thread 的，shared memory 是 per-block 的**，它们的生命周期绑定在
  当前 kernel 的执行期间。kernel 里所有 block 执行完、grid 退出后，SM 会立刻回收这些
  片上资源交给下一个要跑的 kernel/block 使用——上一个 kernel 算出来的值早就被覆盖了。
- **Block 到 SM 的映射不跨 kernel 保持一致**。即使数据没被覆盖，kernel1 里"第 5 个
  block 算出的 mean"存在 SM #3 的 shared memory 里，也不能保证 kernel2 里"负责同一行
  数据的 block"还会被硬件调度到 SM #3——GPU 的调度是动态的，两次 launch 之间没有这种
  绑定关系。
- 所以"传指针"这个操作本身没问题（kernel2 确实是拿着一个指向 mean/var 的指针去读的），
  关键是这个指针指向哪里。GPU 存储层级里，从"快而不持久"到"慢而持久"依次是
  register → shared memory → L2 cache → global memory（HBM），只有 global memory
  保证在 kernel 边界之外依然存在、且地址空间被两次 launch 共享。中间结果落到 global
  memory 之后，下一个 kernel 再通过指针读回来——这一步的写回 + 读取往返，就是 naive
  多 kernel 版本比 fused 版本慢的根本原因。

Fused kernel 之所以能避开这个问题，是因为 mean/var 算出来之后**同一个 kernel 还没
退出**，同一批 thread/block 仍然存活，寄存器/shared memory 里的值还在，可以直接拿来做
归一化，完全不需要经过 global memory 这一趟。这也是为什么 LayerNorm backward（计算
`dgamma`/`dbeta` 时需要跨行的全局 reduction）有时仍然需要拆成两个 kernel——因为那个
reduction 维度本身就无法在一个 block 内完成，必须经过一轮真正的跨 block 全局同步，
而 global memory 正是唯一能承担这个同步职责的存储。

### 延伸：一个 kernel 如何占满多个 SM，以及 block 间通信为什么必须走 global memory

**Kernel launch 的层级结构**：

```
Kernel launch
  └── Grid（若干个 block）
        └── Block（若干个 thread）
              └── Warp（32 个 thread，硬件调度最小单位）
```

硬件调度器（GigaThread Engine）把 grid 里的 block 分发到 GPU 上所有可用的 SM——**一个
block 只会整体分配给一个 SM，不会被拆开跨 SM 执行**（block 内的 thread 要共享 shared
memory、要做 `__syncthreads()` 同步，这要求它们必须在同一个 SM 上）。但**一个 SM 可以
同时驻留多个 block**，且一个 kernel 通常包含远多于 SM 数量的 block，调度器会把这些
block 铺满所有 SM，跑完一批再调度下一批。这就是"一个 kernel 占用多个 SM"的直接原因：
只要 grid 里的 block 数够多、彼此没有依赖关系，调度器自然会把它们分布到全部 SM 上并行
跑。3.3 节的 `layernorm_kernel` 里 `grid = (n_rows,)`——每一行对应一个 block/program，
行数远多于 SM 数时，这些 program 会分散到所有 SM 上同时执行，这才是 LayerNorm 能跑得
快的原因：不是单个 SM 算得快，是几千个 block 铺在上百个 SM 上一起算。

决定"占用多少个 SM"的因素：
1. **grid 里 block 的总数**——block 数远多于 SM 数才能铺满；如果 grid 只有 1 个 block，
   这个 kernel 就只会用到 1 个 SM，其余全部闲置
2. **每个 block 占用的资源（register、shared memory、thread 数）**——决定同一个 SM
   能同时驻留几个 block（occupancy），但只要 block 总数够多，仍会分布到所有 SM，只是
   每个 SM 同时活跃的 block 少一些
3. **block 之间是否天然独立、无依赖关系**——这是能否被自由调度、铺满所有 SM 的前提

**Block 间通信为什么必须走 global memory**：block 是软件概念，SM 是硬件执行单元，
调度器把每个 block 整体分配给某个 SM——所以"block 间数据交换"本质上就是"SM 间数据
交换"的软件说法。完整的通信能力分三层：

```
同一 block 内的 thread   → 可用 shared memory 通信 + __syncthreads() 同步
同一 SM 上的不同 block   → 不能直接通信（CUDA 编程模型强制隔离各 block 的 shared memory）
不同 SM 上的 block       → 不能直接通信（硬件上 SM 之间没有直连通路）
```

关键点：**只要跨了 block 边界，不管两个 block 是否恰好被调度到同一个 SM 上，通信方式
都一样——都必须走 global memory（或 atomic 操作、cooperative groups 的 grid 同步），
没有"物理上挨着就能抄近路"这种优化。** 这是 CUDA 编程模型故意做的隔离设计：block 的
执行顺序和位置不保证（调度器可能先跑 block 0 和 block 5，block 3 要等某个 SM 空出来
才轮到），如果允许 block 间任意通信，会有死锁风险，也会限制硬件扩展到成百上千 SM 的
能力。跨 block 通信的正规通道只有三种：全局 `atomicAdd` 之类的原子操作、cooperative
groups 的 grid-level 同步（要求所有 block 同时驻留，代价高）、或者干脆拆成两个 kernel
用 global memory 交接（LayerNorm backward 算 `dgamma`/`dbeta` 时的跨行 reduction 就是
典型例子）。

### 实际运行

```bash
python3 code/03_fused_layernorm.py
```

输出：

```
torch      : 2.9.1
triton     : 3.5.0
cuda avail : True
device     : NVIDIA Thor
capability : (11, 0)
------------------------------------------------------------
[PASS] fused_layernorm: allclose=True, max_abs_err=1.907e-06
triton fused layernorm : 0.0919 ms
naive multi-kernel     : 0.4761 ms
torch  F.layer_norm    : 0.1089 ms
```

这次结果很清楚：融合后的 Triton kernel（0.09 ms）比手动拆成多步的 naive 实现
（0.48 ms）快了 **5.2 倍**，甚至比 PyTorch 内置的 `F.layer_norm`（0.11 ms，本身也是
融合过的 kernel）还快一点。这说明了 Triton 的核心价值：**当你能比通用库更精确地知道
自己的融合边界时，手写 kernel 能拿到实打实的收益**——而 vector add/softmax 这类已经是
单一 kernel 的简单算子，手写和调用库基本没有差异（对应 2.2/2.3 节的结果）。

## 3.4 Profiling：`triton.testing.do_bench` 与查看生成的 PTX

完整代码见 [`code/03_profiling.py`](code/03_profiling.py)。

```python
import triton.testing
ms = triton.testing.do_bench(lambda: triton_add(x, y))
```

`do_bench` 是 Triton 自带的计时工具（内部用 CUDA event，做 warmup + 多次重复取中位数），
和我们在 `common.py` 里自己写的 `bench_ms` 是同一类工具，直接用官方提供的即可。

查看某个已编译 kernel 的 PTX（Triton 3.5 的内部 API，通过 `JITFunction.device_caches`）：

```python
device_id = next(iter(add_kernel.device_caches))
kernels_by_signature = add_kernel.device_caches[device_id][0]
compiled_kernel = next(iter(kernels_by_signature.values()))
ptx = compiled_kernel.asm["ptx"]
```

### 实际运行

```bash
python3 code/03_profiling.py
```

输出：

```
do_bench vector_add: 0.1204 ms
do_bench torch add : 0.1189 ms

Generated PTX for add_kernel (first 15 lines of 188):
  //
  // Generated by LLVM NVPTX Back-End
  //

  .version 9.0
  .target sm_110a
  .address_size 64

  	// .globl	add_kernel              // -- Begin function add_kernel
                                          // @add_kernel
  .visible .entry add_kernel(
  	.param .u64 .ptr .global .align 1 add_kernel_param_0,
  	.param .u64 .ptr .global .align 1 add_kernel_param_1,
  	.param .u64 .ptr .global .align 1 add_kernel_param_2,
  	.param .u32 add_kernel_param_3,

Available compiled artifacts: ['source', 'ttir', 'ttgir', 'llir', 'ptx', 'cubin']
```

`.target sm_110a` 确认了生成的 PTX 确实是针对当前 GPU 架构的。`asm` 字典里的 `ttir`/`ttgir`
分别是 Triton IR 和 TritonGPU IR（编译过程中的中间表示），`llir` 是 LLVM IR，一路到 `ptx`/`cubin`——
这条流水线和 `nvcc` 编译 CUDA C++（.cu → PTX → cubin）在终点上是一致的，只是起点从 CUDA C++
换成了 Triton DSL。

### 编译流水线：`asm` 字典里 6 个 artifact 分别是什么

`compiled_kernel.asm` 里的 6 个 key 不是"PTX 的不同版本"，而是**编译流水线上依次不同的
表示形式**，每一步都把上一步"翻译/降级（lower）"成更接近硬件的形式，直到变成真正能跑
的机器码：

```
source (Python/Triton DSL)
   ↓  Triton 前端解析
ttir   (Triton IR)
   ↓  引入硬件相关的 layout/tiling 决策
ttgir  (TritonGPU IR)
   ↓  走 LLVM 后端
llir   (LLVM IR)
   ↓  LLVM 的 NVPTX backend 生成
ptx    (PTX 汇编，文本形式，人可读)
   ↓  ptxas（PTX assembler）做最终编译
cubin  (二进制机器码，真正跑在 GPU 上的东西)
```

逐层含义：

- **`source`**：`@triton.jit` 装饰的 Python 函数源码本身，最原始的形式。
- **`ttir`（Triton IR）**：Triton 解析源码后得到的第一层 IR，还比较高层、接近你写的
  逻辑——`tl.load`/`tl.store`/`tl.dot` 这些操作基本保持原样，还没涉及"具体怎么映射到
  硬件线程"的细节。
- **`ttgir`（TritonGPU IR）**：在 ttir 基础上加入 GPU 执行相关的**具体决策**——数据在
  warp/thread 之间怎么切分（layout）、`tl.dot` 怎么映射到 tensor core 指令等。autotune
  选出的 `num_warps`、tile 大小等参数，就是在生成这一层时体现出来的。
- **`llir`（LLVM IR）**：Triton 把 ttgir 再往下翻译成 LLVM IR——业界通用的编译器中间
  语言，借助 LLVM 现成的通用优化能力和多硬件后端支持（LLVM 有专门针对 NVIDIA GPU 的
  "NVPTX backend"）。
- **`ptx`**：LLVM 的 NVPTX backend 把 llir 翻译成的 PTX 汇编，即前面看到的那份文本。
  **PTX 本身还不是最终机器码**，而是一种"给 NVIDIA GPU 用的、跨代次兼容的中间汇编格式"——
  文本可读，类似 x86 汇编但是给 GPU 架构设计的虚拟指令集，靠驱动 JIT 编译到具体硬件的
  SASS，从而做到同一份 PTX 能在不同代次 GPU 上运行。
- **`cubin`**：PTX 再经过 `ptxas` 编译出来的**二进制机器码**（容器格式叫 cubin，里面
  装的是 SASS）。**这才是实际被 GPU 执行单元直接执行的形式**，是流水线的终点。

一句话理清关系：`ttir`/`ttgir`/`llir` 不是"PTX 的编译产物"，而是**比 PTX 更早、更高层**
的中间表示，是"从 Python 代码走到 PTX"这条路上的中间站；`ptx` 是流水线倒数第二步产出的
文本形式虚拟指令集；`cubin` 才是 PTX 再编译一步后得到的最终二进制机器码。这几层可以用来
做**分层调试**——如果生成的 cubin 性能不对，可以往上查 ttgir 看 layout 选择是否合理，
往下查 ptx/cubin 看具体指令是否用上了预期的硬件特性（tensor core 指令、向量化
load/store 等），这跟排查 C++ 编译优化问题时看 `-emit-llvm` 的 IR 是同一套思路，只是
Triton 把这条流水线的每一层都主动暴露给你了。

如果需要更细粒度的硬件级 profiling（warp occupancy、内存带宽利用率等），本机也装有
`nsys`（Nsight Systems，做时间线级 profiling）和 `ncu`（Nsight Compute，做单 kernel 深度分析），
用法与分析 CUDA C++/cuBLAS/TensorRT kernel 完全一样：

```bash
nsys profile -o report python3 code/02_vector_add.py
ncu --set full -o report python3 code/02_vector_add.py
```

（本教程不展开 nsys/ncu 报告的详细读法，因为你已经熟悉 CUDA/TensorRT 的 profiling 工作流，
Triton 生成的 kernel 在 nsys/ncu 里的呈现方式和普通 CUDA kernel 一致，没有额外的特殊性。）

## 3.5 小结

- Autotune 把"给不同 tile 配置做 benchmark 选最优"这件 TensorRT builder 会做的事，搬到了
  源码层面，由你定义候选空间
- `tl.dot` 让矩阵乘法自动使用张量核心，但教学级实现远不如 cuBLAS——Triton matmul 的真正价值
  在于融合了额外计算的变种（如 flash attention），不是替代通用 GEMM 库
- Kernel 融合（fused LayerNorm）是 Triton 最直接体现价值的场景：本机实测融合版比朴素多
  kernel 版快 5.2 倍
- Triton 编译到 PTX/cubin 的路径与 CUDA C++/nvcc 一致，用你已经会用的 nsys/ncu 分析没有障碍

下一章开始 torch.compile：[第 4 章：torch.compile 基础](04_torch_compile_basics.md)
