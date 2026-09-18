# 第 0 章：全局地图 —— Triton / torch.compile 与你已知技术栈的关系

> 本教程假设你已经掌握 **TorchScript**、**ONNX**、**CUDA C++**、**TensorRT**。
> 这一章的目标不是教语法，而是先在你已有的知识地图上标出 Triton 和
> torch.compile 的位置，后面每一章的例子都会印证这里的类比。

## 0.1 一张对比表

| 维度 | CUDA C++ | TensorRT | TorchScript | **Triton** | **torch.compile** |
|---|---|---|---|---|---|
| 抽象层级 | 最底层，手动管理线程/内存 | 图级 IR + 内置/插件算子 | Python 子集 → 静态图 IR | 块（block）级 DSL，隐藏线程细节 | Python 函数 → 多级 IR → 生成代码 |
| 编译时机 | AOT（Ahead-of-time）（nvcc，构建时） | **AOT**（build engine，离线） | AOT（trace/script 时） | **JIT**（首次调用时，按 shape/dtype 特化，带缓存） | **JIT**（首次调用时，按 shape/dtype 特化，带缓存） |
| 图/代码捕获方式 | 你自己写 kernel，没有"图" | 从 ONNX/API 构建的算子图 | `trace`（跑一遍录制）或 `script`（静态分析 Python AST） | 没有图捕获，你写的就是最终 kernel 逻辑 | **Dynamo**：Python 字节码级捕获，遇到不支持的代码自动"graph break"退回 eager |
| 算子粒度 | 你决定 | 层/算子融合由 builder 决定 | 算子级（aten op） | **你手写的整块计算**（可以是好几个算子融合在一起） | 由 **Inductor** 后端自动做算子融合、决定 kernel 边界 |
| 生成的底层代码 | 你写的 CUDA C++ → PTX/cubin | 内部 kernel（黑盒）/ 插件 | 走 PyTorch 已有 aten kernel，不生成新代码 | Triton IR → LLVM IR → **PTX/cubin**（你能看到全过程） | GPU 上默认生成 **Triton kernel**（本质是调用 Triton 编译器） |
| 可移植性 | 需要为不同架构重新编译 | NVIDIA GPU only，engine 与硬件/版本强绑定 | 较好，随 PyTorch 版本走 | 支持 NVIDIA/AMD 等后端，同一份 Python 代码 | 好，backend 可插拔（Inductor 默认，也可换其他 backend） |
| 调试难度 | 高（gdb-cuda/nsight） | 高（引擎是黑盒，只能看 profiling） | 中（trace 会静默出错，script 报错但信息有限） | 中（Python 报错+能读生成的 PTX，见第 3 章） | 中（`TORCH_LOGS` 系列开关能看到图、guard、生成代码，见第 7 章） |
| 典型使用场景 | 自定义算子、极限性能 | 生产推理部署（C++/多语言服务） | 早期 PyTorch 部署方案（渐被 export/AOTInductor 取代） | 写自定义 GPU kernel（比 CUDA C++ 简单），或研究 torch.compile 生成的代码 | Python 训练/推理加速，几乎零代码改动 |

Dynamo 抓图 → Inductor 在图上做融合决策、切出 kernel 边界 → 每个 kernel 边界生成一份 Triton 源码 → Triton 编译器把它编译成 PTX/cubin；AOTInductor 则是把这整条链路挪到部署前离线跑一次，产出不依赖 Python 的可执行产物，是 TorchScript 之后新的部署路线。

  CUDA C++ → PTX → cubin 这条路你已经走过；Triton 的编译流程最后也落到这两级（Triton IR → LLVM IR → PTX → cubin），区别是你能在 Triton 里主动 dump 出中间产物看，第 3章会演示），而 TensorRT engine 内部生成的 kernel 是黑盒，你只能 profiling，看不到源码级的东西。
  - PTX：NVIDIA GPU 的虚拟指令集（类似 CPU 世界的 LLVM IR），是架构无关的中间表示，人可读。
  - cubin：PTX 被 ptxas 进一步编译成的、绑定具体 GPU 架构（sm_XX）的二进制机器码，GPU 实际执行的是这个。

  块级（block-level）DSL 怎么理解
  - CUDA C++ 是线程级编程模型：你写的 kernel body 描述单个线程做什么，threadIdx/blockIdx 要自己算索引，跨线程通信（shared memory、reduction）要自己手写同步。
  Triton 是block/tile 级编程模型：@triton.jit 函数里你描述的是"一个 block 处理一整块数据"——tl.arange 直接给你一个 block 内的一批索引（一个向量/矩阵而不是一个标量），tl.load/tl.store 一次搬一整块数据（编译器自动做向量化访存、合并访问），reduction 用 tl.sum/tl.max 这类原语，编译器负责把这些"块级操作"编译成底层真正的线程级并行代码（怎么分线程、怎么用 shared memory）。
  - :Triton 相当于给你一层"SIMD-over-block"的抽象，你写的代码看起来像在操作 numpy 数组（块状语义），但背后编译器自动映射到 CUDA 的线程网格。这就是为什么文档说它"隐藏线程细节"。

  TorchDynamo 是 torch.compile 的图捕获前端，工作在 Python 字节码层面（不是像 torch.jit.trace 那样跑一遍记录 tensor 操作，也不是像 torch.jit.script 那样静态分析 AST）：
  - 它挂一个 frame evaluation hook，在函数真正执行前拦截 Python 字节码，尝试把里面的 tensor 操作转成一张 FX Graph。
  - 遇到它无法安全转换的 Python 结构（任意的第三方库调用、复杂控制流、print 之类的副作用等），它就在那里切一刀——graph 
    break：这段之前捕获的代码变成一张编译好的图，中间这一小段退回 CPython eager 执行，之后再继续尝试捕获下一段。
  - 为了保证正确性（不像 trace 会静默出错），Dynamo 会生成 guard（对输入 shape/dtype/某些 Python 值的假设条件），下次调用先检查 guard
    是否仍然成立，不成立就重新触发编译——这就是文档里"按 shape/dtype 特化，带缓存"的机制来源。

  Inductor 自动算子融合 / 决定 kernel 边界，怎么理解
  发生在更细的算子图层面
  1. 输入：AOTAutograd 给 Inductor 的是一张 FX Graph，节点是 aten 算子（比如 add、relu、mul、sum 等），逐个算子之间默认是要读写显存的（每个算子输出一次中间 tensor）。
  2. 融合决策：Inductor 做的核心事情是分析这些算子之间的数据依赖 + 访存模式（是不是 elementwise、是不是能共享同一套循环/索引、有没有 reduction 打断融合），把能合并进"同一个 kernel里一次性算完"的相邻算子分到一组——这就是"决定 kernel 边界"：一条边界内的多个 aten 算子最终变成一个 Triton kernel，中间结果留在寄存器/shared memory 里，不写回 HBM。这直接对应你在 CUDA 里手写 fused kernel 想达到的效果（省掉中间 tensor 的显存读写往返），只是这里是编译器自动做的，不需要你手写。
  3. 典型可融合的模式：pointwise 算子链（比如 x + y 后接 relu）几乎总能融合；reduction（sum/softmax 里的 max/sum）会成为一个融合边界的"锚点"，前后的 pointwise 操作可以融合进同一个 kernel 但 reduction 本身决定了这个kernel 的循环结构；matmul 这类有专门 tuned kernel（模板/backend 选择，如 Triton matmul 模板或直接调 cuBLAS）的算子通常自成一个 kernel，不会被随意融合，但它前后的 elementwise（比如 bias add + activation）常被融合进它的 epilogue。
  4. 决策依据：Inductor 内部用 scheduler，基于成本模型（估算融合前后的显存访问量、并行度）和一些启发式规则来决定分组；你不需要手动指定，但能看到结果——TORCH_LOGS=output_code 会把它最终生成的 Triton kernel 源码打出来，第4 章会让你实测对照"哪几个 aten 算子被合进了同一个 triton_poi_fused_xxx kernel"。

## 0.2 architecture：torch.compile 内部流水线

```
你的 nn.Module / 函数
        │
        ▼
┌───────────────────┐
│   TorchDynamo      │  字节码级图捕获（类比：比 TorchScript 的 trace/script 更宽容）
│  (graph capture)   │  遇到不支持的 Python 结构 → graph break，退回 eager 跑那一小段，之后继续捕获
└─────────┬──────────┘
          │ FX Graph（前向）
          ▼
┌───────────────────┐
│   AOTAutograd      │  在 FX Graph 基础上生成前向 + 反向图（训练场景需要）
└─────────┬──────────┘
          │ 优化后的 FX Graph
          ▼
┌───────────────────┐
│    Inductor        │  后端 codegen：决定算子融合边界、tiling、内存分配
│   (backend)        │  → GPU: 生成 **Triton kernel** 源码并调用 Triton 编译
│                     │  → CPU: 生成 C++/OpenMP 代码
└─────────┬──────────┘
          │ Triton kernel 源码（GPU 路径）
          ▼
┌───────────────────┐
│   Triton 编译器     │  Triton IR → TritonGPU IR → LLVM IR → **PTX** → **cubin**
└────────────────────┘
```

对照你熟悉的 TensorRT 流程：

```
ONNX 图 → TensorRT builder（离线，AOT）→ 层融合 + kernel/tactic 自动选型 → engine（cubin 级黑盒）
```

两条流水线在"图 → 融合决策 → 生成/选择底层 kernel → 编译到 cubin"这个骨架上是相通的，
核心区别是 **TensorRT 在部署前离线构建**（需要预先固定/给定输入形状范围），
**torch.compile 在第一次真实调用时动态构建**（用 guard 机制检测输入变化，必要时自动重新编译，见第 5 章）。

## 0.3 三个关键类比，贯穿全书

**类比 1：Triton 是"块级 CUDA"**
写 CUDA C++ 时你要手动算 `blockIdx.x * blockDim.x + threadIdx.x`、手动管理 shared memory。
Triton 里一个 `@triton.jit` 函数处理的是一整块（tile）数据，`tl.load`/`tl.store` 自动处理向量化访存，
`tl.arange` 直接给你一整块的索引。第 2 章的 vector add 例子会让你直接对比感受这个差异。

**类比 2：torch.compile 是"更宽容的 TorchScript"**
`torch.jit.trace` 录制一次执行路径，如果模型有依赖输入的分支（`if x.sum() > 0`），trace 会
**静默地**把某一次跑到的分支烙死进图里，出了训练/推理输入分布变化的 bug 很难查。
`torch.jit.script` 通过静态分析 Python AST 支持控制流，但对 Python 语法的支持有限，很多写法会直接报错。
torch.compile 的 Dynamo 用**字节码级**捕获，遇到实在无法追踪的代码就"打断图"（graph break），
用 eager 跑那一小段，跑完继续追踪 —— 结果总是正确的，只是可能没有完全融合。第 4、5 章会实测演示。

**类比 3：Triton 和 torch.compile 是"手动挡"与"自动挡"**
torch.compile 的 Inductor 后端在 GPU 上生成的就是 Triton kernel —— 学会读 Triton 代码，
就能看懂 `TORCH_LOGS=output_code` 吐出来的东西（第 4 章会实测验证）。
反过来，当 Inductor 自动生成的 kernel 性能不够、或者你需要某种它不会自动生成的融合模式时，
可以自己写 Triton kernel，再通过 `torch.library.triton_op` 把它注册成一个自定义算子接入
torch.compile 的图（第 5 章的重点例子）—— 这类似于给 TensorRT 写自定义 plugin，
但整个过程都在 Python 里，且能被 torch.compile 继续融合周围的算子。
- Inductor 是什么：是 torch.compile 流水线里的代码生成后端（codegen backend）——Dynamo 抓图、AOTAutograd 生成前向+反向图之后，Inductor 接手最后一步：把这张 FX Graph（节点是 aten 算子）编译成真正能跑在硬件上的代码。torch.compile 架构上是前端（Dynamo，负责图捕获）+ 后端（可插拔，负责把图变成可执行代码）分离的，Inductor 是默认的后端，但理论上可以换别的后端（比如早期的 nvfuser、或者你自己注册的 backend）。类比 ONNX 2engine：ONNX 图之后交给 tensorrt 做层融合+选 kernel，这个 tensorrt 就是那个"后端builder"；Inductor 在 torch.compile 里扮演的正是这个角色。
- Inductor 自己不直接写 PTX，它是把"要生成什么样的 GPU kernel"这件事翻译成 Triton 代码，复用 Triton 编译器做最后一段。
- Dynamo 负责"看懂你的 Python 代码在做什么"，Inductor 负责"决定怎么把这些操作高效地编译成 GPU/CPU 代码"——它是 torch.compile 里真正做"融合 + codegen"这个脏活的组件，角色上等价于 TensorRT 的builder，但产出是可读的 Triton/C++ 源码而不是黑盒 engine。


## 0.4 本教程的实测环境

本教程所有代码在以下环境**实际运行验证**，文档中的延迟数字都是真实测得，不是估算：

- 硬件：NVIDIA Thor (Jetson, aarch64), compute capability (11, 0) / sm_110a
- CUDA 13.0，驱动配套的 `ptxas` 版本 13.0
- PyTorch 2.9.1，Triton 3.5.0（版本选择原因见 [01_setup.md](01_setup.md)）
- TensorRT 10.14.1（用于第 6 章对比）

## 0.5 章节路线图

1. [环境搭建](01_setup.md) —— 安装 Triton，解决新架构下的编译问题
2. [Triton 基础](02_triton_basics.md) —— vector add、softmax
3. [Triton 进阶](03_triton_intermediate.md) —— autotune、matmul、fused layernorm、profiling
4. [torch.compile 基础](04_torch_compile_basics.md) —— Dynamo/AOTAutograd/Inductor、查看生成代码
5. [torch.compile 进阶](05_torch_compile_advanced.md) —— graph break、动态 shape、接入自定义 Triton kernel
6. [综合对比](06_benchmark_comparison.md) —— 五种路径的真实 benchmark
7. [附录](07_appendix.md) —— 调试技巧、选型指南、常见坑
