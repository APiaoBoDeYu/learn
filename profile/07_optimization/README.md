# 第 07 章 常见优化手段（与瓶颈一一对应）

> 📎 对应笔记：[doc/CUDA.xmind](../doc/CUDA.xmind) →「四大瓶颈」的修复方案、「使用 CUDA 进行预处理/后处理」、「host-side 内存操作与 CUDA Graph」；[doc/TensorRT_副本.xmind](../doc/TensorRT_副本.xmind) →「关键优化技巧」
>
> 前面各章回答的是"瓶颈在哪里"，这一章回答"用什么手段、能快多少、什么时候没用"。每个手段都给出本机上**最小的前后对比**，已在前面章节测过的直接引用。最重要的一列是最后一列：**同一个手段用错了地方，收益为 0 甚至为负**。

## 学习目标

1. 记住"瓶颈 → 手段"的对应关系，以及每个手段在 Thor 上的实测收益量级；
2. 知道哪些手段对哪类瓶颈**无效**（CUDA Graph 对 Memory-bound、提高 Occupancy 对低 ILP kernel、compile 对小模型……）；
3. 掌握 VLA / LLM 推理的特殊性：prefill 和 decode 是两类完全不同的负载；
4. 养成"度量 → 定位 → 假设 → 修改 → 验证"的闭环习惯，每一步都和基线对比。

## 目录

- [7.0 总表：瓶颈 → 手段 → 实测收益](#70-总表瓶颈--手段--实测收益)
- [7.1 CPU 侧：预处理与数据加载](#71-cpu-侧预处理与数据加载)
- [7.2 同步点与 Launch-bound](#72-同步点与-launch-bound)
- [7.3 Memory-bound：融合、低精度、数据排布、FlashAttention](#73-memory-bound融合低精度数据排布flashattention)
- [7.4 显存分配与动态 shape](#74-显存分配与动态-shape)
- [7.5 VLA / LLM 推理专项：prefill vs decode](#75-vla--llm-推理专项prefill-vs-decode)
- [7.6 优化闭环与负优化案例](#76-优化闭环与负优化案例)
- [常见坑](#常见坑) · [练习](exercises.md) · [小结](#小结)

| Demo | 内容 |
|---|---|
| [01_cpu_side.py](01_cpu_side.py) | 预处理：Python 循环 / numpy / GPU；DataLoader 的 num_workers 与 pin_memory |
| [02_sync_launch.py](02_sync_launch.py) | `.item()` 的真实代价；ManySmallOps 的 eager / CUDA Graph / compile；同样的手段用在 Memory-bound、Compute-bound 上 |
| [03_memory_bound.py](03_memory_bound.py) | fp16/bf16、channels_last、朴素 attention vs SDPA |
| [04_alloc_dynamic.py](04_alloc_dynamic.py) | allocator（默认 / expandable_segments / cudaMallocAsync / 预分配）；动态 shape 与 torch.compile（冷/热缓存） |
| [05_llm_decode.py](05_llm_decode.py) | GPT 式小模型：prefill vs decode、KV cache、batch、decode 的 CUDA Graph |

---

## 7.0 总表：瓶颈 → 手段 → 实测收益

（"收益"均为本机实测的加速比；⬇ 表示负优化。）

| 瓶颈 | 诊断 | 手段 | 实测收益 | 无效 / 负优化的情况 |
|---|---|---|---|---|
| Python/CPU 开销 | VizTracer、py-spy（第 02 章） | 向量化 | 预处理 2.13x（7.1） | |
| | | 预处理放到 GPU | 4.49x（只把归一化放到 GPU）/ 11.87x（resize 也放到 GPU）（7.1） | A4 与 CPU 结果差 1.3e-2（插值实现不同），要确认精度可以接受 |
| 数据加载 | nsys、py-spy | DataLoader `num_workers` | 8 个 worker：340 → 3804 样本/秒（11x）（7.1） | |
| | | `pin_memory` | +0~3%（7.1）；流水线的 no_pin 开关约 1.03x（T01） | Thor 是统一内存，pinned 的收益远小于独显平台 |
| 同步点 | torch.profiler、nsys `--cudabacktrace`（第 03/04 章） | 去掉循环中的 `.item()`，推迟到真正需要结果时 | GEMM + CPU 工作：1.10x（省掉了 2 ms 的串行等待）；后处理 5.0x（7.2） | 同步本身只有几十微秒，**代价是 CPU/GPU 无法并行**；GPU 空闲时去掉它几乎没有收益 |
| Launch-bound | nsys 看 kernel 间隙（第 04 章） | CUDA Graph | ManySmallOps 6.38x（7.2）；ToyVLA 1.98x（第 04 章）；TensorRT ToyVLA 1.39x（第 06 章） | **Memory-bound 1.02x、GEMM 1.00x**（7.2） |
| | | torch.compile（融合） | ManySmallOps 33.8x（7.2）；ToyVLA 1.51x（第 03 章） | 小 MLP + 动态 shape：比 eager 慢 1.9 倍 ⬇（7.4） |
| | | compile reduce-overhead | ToyVLA 2.96x（第 03 章） | 已经融合成 1 个 kernel 时比普通 compile 慢（0.069 vs 0.049 ms）⬇（7.2） |
| | | TensorRT | ToyVLA fp16 + Graph 10.3x vs eager fp32（第 06 章） | |
| 拷贝/计算没有重叠 | nsys + GPU metrics（第 04 章 4.8） | 多 stream、双缓冲 | **Thor 上无效**：≥256 KB 的拷贝由 SM 执行，和 kernel 抢 SM | 走 Copy Engine 的小拷贝能重叠，但带宽只有约 14 GB/s，总时间反而 ×1.9 ⬇ |
| | | 多 stream 跑占不满 SM 的长 kernel | 6.68x（第 04 章） | 短 kernel：0.78x ⬇（第 04 章） |
| Memory-bound | ncu + DRAM 换算（第 05 章） | 合并访存 | 6.5x（第 05 章） | |
| | | 算子融合 | 15 个逐元素 kernel → 1 个：16.5x（7.2）；3 → 1：2.9x（第 05 章） | ncu 单看每个 kernel 都"满带宽"，必须在算子层面发现 |
| | | 低精度（fp16/bf16） | GELU 2.07x（7.3） | |
| | | channels_last | TinyCNN fp32 1.35x、fp16 1.56x（7.3） | |
| | | FlashAttention（SDPA） | L=4096：12.9x，峰值显存 516 → 4.1 MB（7.3） | L=256 只有 2.6x |
| Compute-bound | ncu Tensor Core 指标（第 05 章 5.8） | TF32 / FP16 / BF16 | GEMM 4096：FP32 6.2 → TF32 52 → FP16 129~145 TFLOPS（第 05 章） | |
| | | 维度对齐到 8 的倍数 | K=4095 → 4096：6.5x（第 05 章） | |
| Register-bound | ncu Occupancy、stall（第 05 章 5.5） | 软件预取（增加 ILP） | 1.62x（Occupancy 反而从 25% 降到 16.7%） | 提高 Occupancy（25% → 83%）：0.99x；`__maxnreg__(64)`：0.33x ⬇ |
| Control-flow-bound | ncu Active Threads Per Warp | 按 warp 对齐分支 / 排序 | 1.96x / 1.90x（第 05 章） | 排序本身 2.25 ms，比节省的 0.64 ms 还多 ⬇ |
| Bank conflict | ncu wavefront、Source 页 | padding / swizzle | 1.07x（第 05 章） | 收益远小于 ncu 规则预估的 91% |
| 显存分配/碎片 | memory_stats、snapshot（第 03 章） | expandable_segments | reserved 14.6 GB → 1.38 GB，快 1.9x（7.4） | |
| | | 预分配 | reserved 480 MB，快 1.8x（7.4） | |
| 动态 shape | compile 日志、counters | 分桶 padding | 冷启动编译 8 次 → 3 次（6.31 s → 1.78 s）（7.4） | 稳态延迟变长（0.105 → 0.146 ms，padding 多算了）⬇ |
| decode（LLM/VLA） | 7.5 | KV cache | prompt 128：1.28x；prompt 512：2.14x（7.5） | 序列短时收益很小 |
| | | 增大 batch | batch 64：吞吐 10.8x（7.5） | 单个请求的延迟变长 5.9 倍 |

## 7.1 CPU 侧：预处理与数据加载

[01_cpu_side.py](01_cpu_side.py)，日志 [outputs/01_cpu_side.log](outputs/01_cpu_side.log)。

**A. 预处理一帧**（480×640 uint8 → 1×3×224×224 归一化张量，wall 计时，包含 H2D 和 GPU 计算）：

| 方案 | p50 | 加速 | 与 A1 的最大差异 |
|---|---|---|---|
| A1 Python 逐行循环（慢流水线） | 3.105 ms | 1.00x | |
| A2 numpy 向量化 | 1.458 ms | 2.13x | 0 |
| A3 CPU resize + 只拷贝 uint8 + GPU 归一化（快流水线） | 0.691 ms | 4.49x | 1.5e-07 |
| A4 拷贝原始 480×640 uint8，GPU 上 resize + 归一化 | 0.261 ms | 11.87x | 1.3e-02 |

- A3 的 0.691 ms 中有 0.483 ms 是 `cv2.resize`：把归一化挪到 GPU 之后，**新的瓶颈是 CPU 上的 resize**。A4 把 resize 也放到 GPU 上，又快了 2.6 倍。这是"修复一个瓶颈后，瓶颈会转移"的典型例子。
- A4 与 CPU 版本有 1.3e-2 的差异：`F.interpolate(bilinear)` 和 `cv2.resize(INTER_LINEAR)` 的插值实现不同。把预处理换到 GPU 上时，训练和部署要用同一种实现，否则会引入分布偏移。
- 📎 CUDA 笔记中"使用 CUDA 进行预处理/后处理"的手写 kernel 是 A4 的进一步形式（resize + 归一化 + HWC→CHW 融合成一个 kernel），本章没有实现。

**B. DataLoader**（每个样本 JPEG 解码 + resize，单个样本 2.6 ms，batch=8）：

| num_workers | 0 | 2 | 4 | 8 |
|---|---|---|---|---|
| pin_memory=False | 340 | 743 | 1649 | 3709  样本/秒 |
| pin_memory=True | 341 | 764 | 1693 | 3804 样本/秒 |

- 解码是 CPU 密集型操作，worker 数几乎线性提升吞吐（Thor 有 14 个 CPU 核）。
- `pin_memory` 只有 0~3% 的收益：Thor 的 CPU 和 GPU 共享 DRAM，pageable 拷贝本来就不慢（第 04 章 4.8：大块拷贝本身由 SM 执行）。在独显平台上，pinned 内存是异步 H2D 的前提，收益要大得多。
- GPU 解码（NVJPG/NVDEC、DALI）本章未实测。

## 7.2 同步点与 Launch-bound

[02_sync_launch.py](02_sync_launch.py)，日志 [outputs/02_sync_launch.log](outputs/02_sync_launch.log)。

**A1. 同步的真正代价**（fp32 GEMM 4096 约 22.3 ms + 2 ms 的 CPU 工作）：

| 顺序 | p50 |
|---|---|
| GEMM → `.item()` → CPU 工作 | 24.535 ms（= GEMM + CPU，串行） |
| GEMM → CPU 工作 → `.item()` | 22.340 ms（= max(GEMM, CPU)，并行）：**1.10x** |

`.item()` 本身只要几十微秒，它的代价是**让 CPU 停下来等 GPU**，于是本来可以和 GPU 并行的 CPU 工作被串行化。所以修复的原则不是"去掉所有同步"，而是**把同步推迟到真正需要结果的地方**，让 CPU 先把能做的事做完。

**A2. 流水线的后处理**：

| eager_small_ops | sync_in_loop | p50 | 加速 |
|---|---|---|---|
| True | True | 3.574 ms | 1.00x |
| True | False | 3.259 ms | 1.10x |
| False | True | 0.369 ms | 9.69x |
| False | False | 0.074 ms | 48.3x |

小算子（7×16 次索引赋值）是大头，`.item()` 在 eager 版本中只占 10%；但向量化之后，剩下的 7 次 `.item()` 就成了主要开销（0.369 → 0.074 ms，5.0x）。**一个瓶颈的重要程度取决于其他部分有多快**。

**B. Launch-bound：ManySmallOps(200)**（1024 个元素，eager 下 300 个 kernel）：

| 方案 | GPU p50 | 加速 |
|---|---|---|
| eager | 1.663 ms | 1.00x |
| 手动 CUDA Graph | 0.261 ms | 6.38x |
| torch.compile（融合成 1 个 kernel） | 0.049 ms | 33.80x |
| compile reduce-overhead | 0.069 ms | 23.99x |

CUDA Graph 消除了 launch 开销，但 300 个 kernel 仍然各自执行；融合则让它们根本不存在。reduce-overhead 比普通 compile 慢：融合后只剩 1 个 kernel，没有 launch 开销可以省，反而多了把输入拷进静态缓冲区的开销。

**C. 同样的手段用错地方**：

| 负载 | eager | CUDA Graph | torch.compile |
|---|---|---|---|
| Memory-bound：64M 元素，15 个逐元素 kernel | 35.994 ms | 35.285 ms（1.02x） | 2.186 ms（**16.47x**） |
| Compute-bound：GEMM 4096 fp16 | 1.005 ms | 1.006 ms（1.00x） | 1.007 ms（1.00x） |

- Memory-bound：每个 kernel 读写 512 MB，耗时约 2.4 ms，几微秒的 launch 开销可以忽略，所以 CUDA Graph 无效；融合把 15 次读写变成 1 次，所以有 16 倍的收益。
- Compute-bound：cuBLAS 的 GEMM kernel 已经很高效，compile 也只是调用同一个 kernel。要加速它，只能换精度（TF32/FP16/FP8）或减少计算量。

## 7.3 Memory-bound：融合、低精度、数据排布、FlashAttention

[03_memory_bound.py](03_memory_bound.py)，日志 [outputs/03_memory_bound.log](outputs/03_memory_bound.log)。融合见 7.2 C 和第 05 章 5.4。

**低精度**（GELU，64M 个元素）：fp32 2.335 ms（229.9 GB/s）→ fp16 1.127 ms（238.1 GB/s）/ bf16 1.128 ms（238.0 GB/s），**2.07x**。三者的带宽都接近 240 GB/s 的峰值，所以时间完全由字节数决定：Memory-bound 算子的耗时 ≈ 字节数 / 带宽。

**channels_last**（TinyCNN batch=8，eager）：

| | NCHW | NHWC（channels_last） | 加速 |
|---|---|---|---|
| fp32 | 2.736 ms | 2.024 ms | 1.35x |
| fp16 | 1.846 ms | 1.186 ms | 1.56x |

cuDNN 的 Tensor Core 卷积 kernel 使用 NHWC 布局，输入是 NCHW 时要插入 `nchwToNhwcKernel` 等布局转换（第 04 章 nsys 中占 6.9%；第 05 章 ncu 中占 8.5%）。fp16 + channels_last 合计是 fp32 NCHW 的 2.31 倍。TensorRT 中对应的是 `__myl_CastConc` 这类转换层（第 06 章 6.3）。

**FlashAttention**（fp16，B=1，H=8，D=64）：

| L | 朴素 ms | SDPA ms | 加速 | 朴素峰值显存 | SDPA 峰值显存 |
|---|---|---|---|---|---|
| 256 | 0.059 | 0.022 | 2.64x | 2.2 MB | 0.3 MB |
| 1024 | 0.279 | 0.072 | 3.89x | 33.0 MB | 1.0 MB |
| 4096 | 10.761 | 0.834 | **12.91x** | **516.0 MB** | **4.1 MB** |

朴素实现要把 H×L×L 的注意力矩阵写到显存里再读回来（L=4096 时 8×4096×4096×2 B = 256 MB，加上 softmax 的输出），是典型的"中间张量"问题。FlashAttention 分块计算，中间结果留在 shared memory 中，流量和显存都和 L 成线性关系。本机 `FLASH_ATTENTION` 后端可用；第 05 章 5.9 中 ToyVLA 的 `flash_fwd_kernel` 就是它。

## 7.4 显存分配与动态 shape

[04_alloc_dynamic.py](04_alloc_dynamic.py)，日志 [outputs/04_alloc_dynamic.log](outputs/04_alloc_dynamic.log)。

**A. 逐步增长的分配**（60 步，第 i 步申请 (i+1)×8 MB 并清零，释放上一次的；同时存活的最多 960 MB）：

| allocator | 总时间 | 峰值 allocated | 峰值 reserved | cudaMalloc 次数 |
|---|---|---|---|---|
| 默认 caching allocator | 178.9 ms | 952 MB | **14652 MB** | 60 |
| `expandable_segments:True` | 95.0 ms | 952 MB | 1380 MB | 23 |
| `backend:cudaMallocAsync` | 104.2 ms | 952 MB | 1472 MB | 0（走驱动的显存池，不计入） |
| 预分配最大尺寸，只取 view | 100.7 ms | 480 MB | 480 MB | 1 |

- 默认分配器下，每个新请求都比缓存中的空闲块大，只能申请新段，旧段不归还：**reserved 是实际需要的 15 倍**（第 03 章的同类实验是 30 倍）。在 Jetson 上，这部分显存 CPU 侧也用不了。
- 三种修复都能把 reserved 降到 1.5 GB 以内，并且因为少了 cudaMalloc，速度也快了约 1.8 倍。在推理服务中，**预分配 + 复用**是最可控的方式（7.5 中的 KV cache 也是预分配到 max_len 的）。
- 这些配置必须在 CUDA 初始化之前通过环境变量设置（脚本中每种配置在单独的子进程中运行）。

**B. 动态 shape + torch.compile**（小 MLP + LayerNorm，fp16，60 次调用，序列长度在 17~256 之间随机，54 种不同长度）：

| 方案 | 编译次数 | 60 次总耗时（冷启动） | 60 次总耗时（热缓存） | 稳态 ms/次 |
|---|---|---|---|---|
| eager | 0 | 0.21 s | 0.22 s | **0.059** |
| compile `dynamic=None`（默认） | 2 | 4.03 s | 1.01 s | 0.113 |
| compile `dynamic=False` | 8 | 6.31 s | 0.65 s | 0.105 |
| compile `dynamic=False` + 分桶到 {64,128,256} | 3 | 1.78 s | 0.26 s | 0.146 |

- 默认设置下，第二次见到新 shape 时 dynamo 会自动把它标记为动态维度，之后只编译一次动态图。
- `dynamic=False` 时每个新 shape 都要重新编译，达到 `recompile_limit = 8` 之后这个函数**静默退回 eager**。
- 分桶把编译次数压到 3 次，但 padding 带来了多余的计算，稳态延迟反而最高。
- **冷启动 vs 热缓存**：inductor 会把编译结果缓存到磁盘（`TORCHINDUCTOR_CACHE_DIR`，默认在 `/tmp`），第二次运行时编译时间降到原来的 1/4 ~ 1/10。部署时可以预热缓存，但要注意容器重启后 `/tmp` 会被清空。
- **这个小模型上，compile 的稳态延迟比 eager 慢 1.9 倍**：每次调用只有 2 次 GEMM，compile 的 guard 检查等开销超过了融合 LayerNorm/GELU 带来的收益。compile 对 ToyVLA 有效（第 03 章 1.51x），对这里的小 MLP 无效，所以每次都要实测。
- TensorRT 处理动态 shape 的方式是 optimization profile（min/opt/max shape），本章未实测。

## 7.5 VLA / LLM 推理专项：prefill vs decode

[05_llm_decode.py](05_llm_decode.py)，日志 [outputs/05_llm_decode.log](outputs/05_llm_decode.log)。模型：GPT 式 decoder，d=1024、16 头、8 层，101.8 M 参数，fp16 权重 204 MB。decode 每一步至少要把全部权重读一遍：204 MB / 240 GB/s = **0.848 ms**，这是 batch=1 时 decode 一步的带宽下限。

> ToyVLA 的动作头一次输出 16 个动作，不做自回归；而真实的 VLA（如以语言模型为 backbone 的模型）在生成文本或动作 token 时，decode 的特性与本节完全相同。

**A. prefill vs decode（batch=1）**：

| | 耗时 | token/s | 算力 | 权重读取带宽 | AI（每读 1 字节权重的 FLOP） |
|---|---|---|---|---|---|
| prefill 512 token | 4.228 ms | 121,107 | 24.66 TFLOPS（峰值的 15.8%） | 48.2 GB/s | ≈ 512 |
| decode 1 token | 2.003 ms | 499 | 0.10 TFLOPS（0.1%） | 101.7 GB/s（峰值的 42%） | ≈ 1 |

- 📎 第 00 章 0.4 节的 Roofline：prefill 位于拐点（约 650 FLOP/B）附近，是 Compute-bound；decode 的 AI ≈ 1，远在拐点左侧，是 **Memory-bound**。两者的优化方向完全相反：prefill 要用好 Tensor Core，decode 要减少字节数（量化权重、增大 batch）。
- decode 只发挥了带宽的 42%：batch=1 的 Linear 实际上是 GEMV，第 00 章实测 fp16 GEMV（冷 L2）也只能达到 108 GB/s（屋顶的 48%），两者吻合。

**B. KV cache**（生成 128 个 token）：

| prompt 长度 | 无 KV cache（每步重算整个序列） | KV cache | 加速 |
|---|---|---|---|
| 128 | 356.1 ms（359 token/s） | 279.2 ms（458 token/s） | 1.28x |
| 512 | 597.8 ms（214 token/s） | 279.6 ms（458 token/s） | 2.14x |

- 有了 KV cache，每步只处理 1 个 token，耗时几乎不随序列长度变化（279 ms ≈ 128 × 2.0 ms + prefill）。
- 序列短时收益很小：重算 128~256 个 token 的成本和 decode 一步差不多，因为两者都受"读一遍权重"的限制。序列越长，KV cache 的收益越大（512 时已是 2.14 倍），代价是显存：本例每个 token 的 KV 是 8 层 × 2 × 1024 × 2 B = 32 KB。

**C. batch**（KV 长度 512）：

| batch | 每步 | token/s | 吞吐（相对 batch=1） |
|---|---|---|---|
| 1 | 2.000 ms | 500 | 1.0x |
| 4 | 2.370 ms | 1,688 | 3.4x |
| 16 | 4.357 ms | 3,672 | 7.3x |
| 64 | 11.852 ms | 5,400 | 10.8x |

权重只读一遍，可以服务 batch 中的所有请求，所以 batch 从 1 增大到 4，每步时间只增加了 19%。这是 LLM 服务做 continuous batching 的原因。代价是单个请求的延迟变长（batch 64 时每步 11.9 ms，是 batch=1 的 5.9 倍）。对机器人这类单请求、延迟敏感的场景，batch 通常只能是 1，此时要靠量化（INT8/INT4/FP8 权重，减少字节数）和投机解码（一次验证多个 token，本章未实测）。

**D. decode 一步用 CUDA Graph**：eager 1.998 ms → Graph 1.795 ms，**只有 1.11x**；权重读取带宽 113 GB/s，离 0.848 ms 的下限还有 2.1 倍。说明 decode 的主要瓶颈不是 launch，而是 GEMV kernel 本身的带宽效率。为了让 decode 能被 capture，KV cache 预先分配到 max_len，写入位置和 attention mask 都是 GPU 上的张量（`index_copy_` + 布尔 mask），这是 CUDA Graph 对"静态地址、静态 shape"的要求（第 04 章 4.6）。

## 7.6 优化闭环与负优化案例

**闭环**：度量（建立基线，锁频、warmup、看分布）→ 定位（漏斗模型，自顶向下）→ 假设（"是 X 导致的，改 Y 能快 Z 倍"）→ 修改（一次只改一处）→ 验证（与基线对比；顺便检查精度）→ 记录。最容易被跳过的是**假设**和**验证**：没有假设就无法判断结果是否合理，没有验证就会把负优化留在代码里。

本教程实测到的负优化（都"看起来应该更快"）：

| 改动 | 预期依据 | 实测 | 出处 |
|---|---|---|---|
| 小 MLP 用 torch.compile | "compile 会融合算子" | 0.059 → 0.113 ms，慢 1.9x | 7.4 |
| 已经融合的模型再用 reduce-overhead | "reduce-overhead 更快" | 0.049 → 0.069 ms | 7.2 |
| 动态 shape 分桶 | "减少重编译" | 稳态 0.105 → 0.146 ms | 7.4 |
| 短 kernel 用 8 个 stream | "并发" | 0.78x | 第 04 章 4.5 |
| 拷贝切成 256 KB 小块和计算重叠 | "双缓冲" | 墙钟 8.39 → 11.15 ms | 第 04 章 4.8 |
| `__maxnreg__(64)` 提高 Occupancy | ncu "TheoreticalOccupancy 预估加速 66.7%" | 0.33x | 第 05 章 5.5 |
| 4 个线程分担状态，Occupancy 25% → 83% | ncu "预估加速 63%" | 0.99x | 第 05 章 5.5 |
| 按 cnt 排序消除 divergence | ncu "ThreadDivergence 预估加速 48%" | kernel 快 1.90x，但排序本身 2.25 ms | 第 05 章 5.6 |

共同点：**工具给出的是"可能的方向"，收益要靠实测确认**。

## 常见坑

1. **只优化最显眼的部分**：修复一个瓶颈后要重新度量，瓶颈会转移（7.1 A3 → resize；7.2 A2 → `.item()`）。
2. **把 CUDA Graph 当作万能加速**：它只消除 launch 开销，对 Memory-bound、Compute-bound 无效（7.2 C）。
3. **以为同步本身很贵**：同步只要几十微秒，代价在于阻止了 CPU/GPU 并行（7.2 A1）。
4. **在 Thor 上照搬独显的经验**：pinned memory、拷贝/计算重叠的收益很小或为负（7.1、第 04 章 4.8）。
5. **compile 的收益看了冷启动或热缓存其中一种**：编译时间差 4~10 倍（7.4）；`dynamic=False` 超过重编译上限后会静默退回 eager。
6. **默认 allocator + 动态 shape**：reserved 可能达到实际需要的 15~30 倍（7.4、第 03 章）。
7. **decode 用 prefill 的思路优化**：decode 是 Memory-bound（AI ≈ 1），算力指标没有意义，要看权重读取带宽（7.5）。
8. **换到 GPU 预处理后不检查数值**：插值实现不同，差异可达 1e-2（7.1）。

## 练习

见 [exercises.md](exercises.md)。

## 小结

- 每类瓶颈都有对应的手段，收益从几个百分点到几十倍不等（7.0 总表），但**同一个手段放到错误的瓶颈上，收益为 0 甚至为负**。
- Thor 特有：统一内存让 pinned memory 和拷贝重叠的价值变小；20 个 SM 让小 batch 的 kernel 更容易填不满（第 05/06 章）；带宽 240 GB/s 决定了 decode 等 Memory-bound 负载的上限。
- VLA/LLM：prefill 是 Compute-bound，decode 是 Memory-bound，要分开度量、分开优化。
- 下一章把这些手段按顺序用到慢流水线上，逐步记录每一步的收益。
