# 第 04 章 系统时间线：Nsight Systems（nsys）

> 📎 对应笔记：[doc/TensorRT_副本.xmind](../doc/TensorRT_副本.xmind) →「Nsight Systems（系统级分析）」「三步法」；[doc/CUDA.xmind](../doc/CUDA.xmind) →「STREAM 和 EVENT」「host-side 内存操作与 CUDA Graph」「四大瓶颈·Launch-bound」
>
> 漏斗模型的第 3 层（L3）："它不关心你执行的是什么层，它关心 **GPU 到底在干什么**。" nsys 同时记录 CPU 侧的 API 调用和 GPU 上的 kernel、拷贝，把两者放在同一条时间线上。**在 Thor 上，它是唯一能拿到 GPU kernel 时间的工具。**

## 学习目标

1. 掌握 nsys 的采集方式（只采稳态、NVTX 标注、各种 `-t` 选项）和开销；
2. 会用 `nsys stats` / `nsys analyze` 在命令行里分析，也会直接用 SQL 查询导出的 sqlite；
3. 从时间线中识别：GPU 空闲（bubble）、Launch-bound、同步点、GPU 工作"溢出"到下一个阶段；
4. 理解 stream 并发和 CUDA Graph 在什么时候有效、什么时候无效（含 Thor 上的实测结果）；
5. 用 `--pytorch=autograd-nvtx` 得到"每个 PyTorch 算子的 GPU 时间"。

## 目录

- [4.1 采集](#41-采集)
- [4.2 命令行分析：stats 与 analyze](#42-命令行分析stats-与-analyze)
- [4.3 用 SQL 分析 + 时间线图](#43-用-sql-分析--时间线图)
- [4.4 每个 PyTorch 算子的 GPU 时间](#44-每个-pytorch-算子的-gpu-时间)
- [4.5 Stream 与 overlap](#45-stream-与-overlap)
- [4.6 CUDA Graph](#46-cuda-graph)
- [4.7 GUI 与其他功能（Thor 上的可用性）](#47-gui-与其他功能thor-上的可用性)
- [4.8 GPU metrics：空间维度的利用率、拷贝由谁执行](#48-gpu-metrics空间维度的利用率拷贝由谁执行)
- [常见坑](#常见坑) · [练习](exercises.md) · [小结](#小结)

| Demo | 内容 |
|---|---|
| [workload_nvtx.py](workload_nvtx.py) | 公共负载：只在稳态时用 `cudaProfilerStart/Stop` 打开采集 |
| [01_nsys_basics.sh](01_nsys_basics.sh) | 采集 + `nsys stats` + `nsys analyze` |
| [02_analyze_sqlite.py](02_analyze_sqlite.py) | SQL 分析：每个阶段的 CPU/GPU 时间、launch 延迟；画时间线图 |
| [03_streams_overlap.py](03_streams_overlap.py) | 拷贝与计算的重叠、多 stream 并发 |
| [04_cuda_graph.py](04_cuda_graph.py) | CUDA Graph：收益、限制、常见错误 |
| [05_pytorch_python.sh](05_pytorch_python.sh) | `--pytorch`、`--python-sampling`、GPU metrics |
| [06_op_gpu_time.py](06_op_gpu_time.py) | 每个 PyTorch 算子的 GPU 时间（替代 torch.profiler 的 CUDA time 列） |
| [07_gpu_metrics.py](07_gpu_metrics.py) | GPU metrics：流水线各阶段的 SM/Tensor 利用率；拷贝由 Copy Engine 还是 SM 执行；nsys 的观察者效应 |

---

## 4.1 采集

```bash
nsys profile \
    -t cuda,nvtx,osrt,cublas,cudnn \          # 追踪哪些 API（-t none 表示只做采样）
    --capture-range=cudaProfilerApi \         # 只采集 cudaProfilerStart/Stop 之间的部分
    --capture-range-end=stop \                # 遇到 Stop 后停止采集，程序继续运行（默认 stop-shutdown 会直接结束程序！）
    --cuda-memory-usage=true \                # 显存使用曲线
    -o outputs/pipeline_slow --force-overwrite true \
    python workload_nvtx.py --steps 30
```

### 只采集稳态部分

```python
for i in range(5): pipe.step(i)       # warmup：import、初始化、cuDNN 选算法都不采集
torch.cuda.profiler.start()           # = cudaProfilerStart
for i in range(3): pipe.step(i)       # 采集刚开始的几步受 nsys 初始化影响，不计时
t0 = time.perf_counter()
for i in range(N):
    with region(f"step_{i}"):         # NVTX 标注（common.region 同时生成 NVTX 和 record_function）
        pipe.step(i)
torch.cuda.synchronize()
ms = (time.perf_counter() - t0) / N   # ⚠️ 必须在 stop() 之前计时
torch.cuda.profiler.stop()            # = cudaProfilerStop，此时 nsys 会把缓冲区落盘（约 1 s）
```

> **实测踩坑**：第一版脚本把计时写在了 `stop()` 之后，测出 nsys 下每步要 58 ms（真实值约 14 ms）。`stop()` 时 nsys 落盘缓冲区的约 1 s 被平摊进了 30 步。我一度以为是 nsys 开销太大，用 sqlite 查了 NVTX 区间之间的空隙才找到原因。

### NVTX 标注

| 方式 | 写法 | 说明 |
|---|---|---|
| PyTorch 内置 | `torch.cuda.nvtx.range_push("x")` / `range_pop()` / `with torch.cuda.nvtx.range("x")` | |
| nvtx 包 | `@nvtx.annotate("x", color="blue")` | 可以设置颜色、domain |
| 本教程 | `with common.region("x")` | 同时生成 NVTX 和 `record_function` |
| 每个算子自动标注 | `torch.autograd.profiler.emit_nvtx()`，或 nsys `--pytorch=autograd-nvtx` | 开销较大，见 4.4 |

### 开销（实测）

| | 不开 nsys | nsys（`-t cuda,nvtx,osrt`） | 开销 |
|---|---|---|---|
| 慢流水线 | 10.72 ms/step | 14.13 ms/step | +32% |
| 快流水线 | 3.24 ms/step | 4.39 ms/step | +35% |
| 单次极小的 kernel launch | 5.0 μs | 6.9 μs | +2 μs/launch |

比 torch.profiler 低一个数量级（torch.profiler 是 3.3~4.9 倍，见第 03 章 3.7）。开销主要来自每次 CUDA API 调用多出的约 2 μs，kernel 越碎，开销占比越大。

> **补充（新容器，2026-11-06）**：慢流水线的开销中有一部分来自 nsys 自身引起的 **pageable 拷贝尖峰**：在 nsys 下，慢版每步一次的 588 KB pageable H2D 拷贝平时约 10 μs，但 33 次中有 9 次要 1~4 ms（最长 4187 μs）；不开 nsys 时直接计时，60 步中 preprocess 最长只有 3.3 ms，没有尖峰。按 9 次 × 约 4 ms / 33 步估算，平均每步约 1 ms，约占慢版 3.4 ms 开销的三分之一（估算）。快版用 pinned 内存，没有这个问题。详见 4.8 D。

## 4.2 命令行分析：stats 与 analyze

[01_nsys_basics.sh](01_nsys_basics.sh) 的完整输出见 [outputs/01_nsys_basics.log](outputs/01_nsys_basics.log)。

### 常用报告

| 报告 | 回答的问题 |
|---|---|
| `cuda_api_sum` | CPU 侧调用了哪些 CUDA API、调用多少次、耗时多少 |
| `cuda_gpu_kern_sum` | GPU 上哪些 kernel 最耗时（加 `:base` 去掉模板参数） |
| `cuda_gpu_mem_time_sum` / `cuda_gpu_mem_size_sum` | 拷贝的耗时和数据量 |
| `nvtx_sum` / `nvtx_pushpop_sum` | 每个 NVTX 区间的 CPU 时间 |
| **`nvtx_gpu_proj_sum`** | 每个 NVTX 区间**投影到 GPU 上**的时间：区间内发起的 GPU 工作从开始到结束跨越了多久 |
| `nvtx_kern_sum` | 每个 NVTX 区间发起了哪些 kernel |
| `cuda_kern_exec_sum` | 每种 kernel 的 launch 耗时、排队时间、执行时间 |
| `osrt_sum` | 系统调用（锁、sleep、IO）的耗时 |

### 慢版的关键输出

```
=== cuda_api_sum ===
 Time (%)  Total Time (ns)  Num Calls  Avg (ns)  ...  Name
     47.6       64,958,964     16,170   4,017.3  ...  cudaLaunchKernel        ← 30 步 16170 次，每步约 490 次
     39.6       54,091,636      4,224  12,805.8  ...  cudaMemcpyAsync
      5.8        7,936,391        528  15,031.0  ...  cudaStreamSynchronize   ← 每步 16 次同步

=== cuda_gpu_kern_sum:base ===
 Time (%)  Total Time (ns)  Instances  Avg (ns)  ...  Name
     44.7       60,086,080        627  95,831.1  ...  Kernel2                        ← fp32 GEMM（cuBLAS）
     19.4       26,073,472      8,712   2,992.8  ...  vectorized_elementwise_kernel  ← 8712 个，中位数 1.3 μs：后处理的小算子
      6.9        9,248,800        561  16,486.3  ...  nchwToNhwcKernel               ← 布局转换！每步 17 次
      5.9        7,957,408        198  40,188.9  ...  sgemm_largek_lds64
      3.4        4,524,288         99  45,699.9  ...  sm80_xmma_fprop_implicit_gemm_tf32f32_..._nhwckrsc_nchw...

=== nvtx_gpu_proj_sum ===
    Range      Total Proj Time  Total Range Time  Instances  Proj Avg (ns)  ...  Avg GPU Ops
 :postprocess      181,424,320       205,324,829         33    5,497,706.7  ...        505.0
 :infer            141,067,296       118,123,531         33    4,274,766.5  ...        168.0
 :preprocess        24,247,520       134,775,209         33      734,773.3  ...          1.0
```

能读出：

1. **fp32 卷积有大量布局转换**：`nchwToNhwcKernel` 每步 17 次，占 GPU 时间的 6.9%。cuDNN 为 fp32 卷积选择了 NHWC 格式的 TF32 kernel（名字里的 `nhwckrsc_nchw`），所以输入要先从 NCHW 转成 NHWC。这正是 📎 TensorRT 笔记中说的"大量 reformat"。fp16 路径下，cuDNN 直接走 channels_last，这类转换就消失了。
2. **infer 的 GPU 投影时间（4.27 ms）比 CPU 时间（3.58 ms）还长**：推理阶段发起的 GPU 工作，在 CPU 已经离开 infer 之后还在继续执行，然后在后处理中被第一个 `.item()` 等掉。第 02 章 2.6 节的异步陷阱，在这里有了直接的证据。
3. **后处理每步有 505 个 GPU 操作**，GPU 投影跨越 5.5 ms，但真正在 GPU 上执行的时间很少（见 4.3）。

### 专家规则（`nsys analyze`）

```bash
nsys analyze -r gpu_time_util,gpu_gaps:gap=0.2,cuda_api_sync,cuda_memcpy_sync,cuda_memcpy_async report.nsys-rep
```

```
** GPU Time Utilization (gpu_time_util):
The following are time regions with an average GPU utilization below 50%%.
 Row#  In-Use (%)  Duration (ns)  Start (ns)    ...
    1        34.4    369,738,468  103,509,988        ← 慢版：GPU 只有 34% 的时间在工作
    1        21.2    143,953,888    8,231,913        ← 快版：GPU 只有 21% 的时间在工作！

** cuda_memcpy_async: (使用 pageable 内存的异步拷贝 → 实际上是同步的)
 Duration (ns)  Src Kind  Dst Kind  Bytes (MB)
     4,187,040  Pageable  Device         0.602        ← 慢版预处理的 H2D 拷贝
```

- **快版的 GPU 利用率比慢版还低**（21% vs 34%）：快版去掉了 GPU 上的低效工作，但 CPU 侧发起 kernel 的速度还是跟不上，GPU 有 79% 的时间在等 CPU。这说明**下一步的优化方向是减少 launch 开销**（CUDA Graph / torch.compile，见 4.6）。
- `gpu_gaps`（≥0.2 ms）没有报告任何空隙：GPU 的空闲是由大量很小的间隙累积起来的，这也是 Launch-bound 的典型特征。
- `gpu_gaps` 默认只报告超过 **500 ms** 的空隙，对推理场景基本没用，要用 `gap=` 参数调小。

## 4.3 用 SQL 分析 + 时间线图

`nsys export -t sqlite` 导出的数据库可以直接用 SQL 查询。常用的表：

| 表 | 内容 |
|---|---|
| `NVTX_EVENTS` | NVTX 区间（`eventType=59` 是 push/pop；名字在 `text` 列或 `StringIds` 表中） |
| `CUPTI_ACTIVITY_KIND_RUNTIME` | CPU 侧的 CUDA API 调用（`nameId` → `StringIds`；`correlationId` 关联到 GPU 操作） |
| `CUPTI_ACTIVITY_KIND_KERNEL` / `_MEMCPY` / `_MEMSET` | GPU 上的操作（start/end/streamId/correlationId） |
| `StringIds` | 字符串表 |

[02_analyze_sqlite.py](02_analyze_sqlite.py) 统计了每个阶段：CPU 时间、该阶段发起的 kernel 的 GPU busy 时间（区间并集）、kernel 数、短 kernel 占比、launch 延迟（`cudaLaunchKernel` 开始 → kernel 开始执行）：

```
=== slow: 30 steps ===
step 平均 14.154 ms，GPU busy 35.6%（kernel/memcpy/memset 区间并集 ÷ 墙钟）
stage         CPU ms GPU busy ms  kernels   <5us  launch→start
preprocess     4.157       0.000        0     0%         0.0us
infer          3.557       3.597      141    53%       259.7us
postprocess    6.216       0.488      378   100%         5.9us

=== fast: 30 steps ===
step 平均 4.349 ms，GPU busy 21.2%（kernel/memcpy/memset 区间并集 ÷ 墙钟）
stage         CPU ms GPU busy ms  kernels   <5us  launch→start
preprocess     0.858       0.018        6   100%         6.7us
infer          3.159       0.889      136    68%         6.5us
postprocess    0.176       0.009        6   100%         6.5us
```

![nsys 时间线](outputs/timeline.png)

（时间线来自 sqlite 数据：每张图是连续的 2 个 step，上面一行是 CPU 侧的 NVTX 阶段，下面一行是 GPU 上的操作。）

**怎么读这张图和表**：

| 现象 | 位置 | 结论 |
|---|---|---|
| 预处理阶段 GPU 一片空白 | 慢版蓝色区间下方 | **CPU-bound**：GPU 在等 Python 预处理（4.2 ms） |
| 推理的 GPU 工作一直延续到后处理阶段 | 慢版橙色区间之后 | GPU 跟不上 CPU（launch→start 延迟 260 μs，kernel 在排队），工作"溢出"到后处理 |
| 后处理阶段 GPU 上全是稀疏的细条 | 慢版绿色区间下方 | **Launch-bound**：378 个 kernel 全部短于 5 μs，CPU 用 6.2 ms 发起，GPU 只忙了 0.49 ms |
| 快版推理阶段 GPU 上很稀疏 | 快版橙色区间下方 | **还是 Launch-bound**：CPU 3.16 ms，GPU 只忙了 0.89 ms；launch→start 只有 6.5 μs，说明 GPU 一直在等 CPU |

> 第 03 章 3.5 节用 `torch.compile(mode="reduce-overhead")` 把 ToyVLA batch=1 的推理压到了 0.79 ms，和这里测得的"GPU 实际工作 0.89 ms"非常接近，两个结果互相印证：**消除 launch 开销后，延迟就逼近 GPU 的纯计算时间**。

## 4.4 每个 PyTorch 算子的 GPU 时间

torch.profiler 在 Thor 上没有 CUDA time 列（第 03 章 3.8），可以用 nsys 替代：

```bash
nsys profile -t cuda,nvtx --pytorch=autograd-nvtx ... -o outputs/pt_ops_slow python workload_nvtx.py --steps 10
nsys export -t sqlite -o outputs/pt_ops_slow.sqlite outputs/pt_ops_slow.nsys-rep
python 06_op_gpu_time.py outputs/pt_ops_slow.sqlite
```

`--pytorch=autograd-nvtx` 会让 nsys 为每个 aten 算子生成一个 NVTX 区间（不需要改代码；`autograd-shapes-nvtx` 还会记录 shape）。[06_op_gpu_time.py](06_op_gpu_time.py) 通过 `correlationId` 找到每个 kernel 的 launch 调用，再找到包住这次 launch 的最内层 aten 区间，按算子名汇总：

```
######## slow
6747 个 kernel，13 个 step，GPU kernel 时间合计 3.994 ms/step
算子（发起 kernel 的最内层 aten 算子）   调用/step kernel/step GPU ms/step   GPU% CPU ms/step
aten::addmm                                     20.0        32.0       1.412  35.4%       0.606
aten::_addmm_activation                          6.0        12.0       0.683  17.1%       0.181
aten::cudnn_convolution                         12.0        31.0       0.679  17.0%       0.488
aten::fill_                                      1.0         1.0       0.424  10.6%       0.025   ← realloc 开关的 zero_()
aten::mul                                      231.0       231.0       0.292   7.3%       2.000   ← CPU:GPU ≈ 7:1
aten::add                                      117.0       117.0       0.155   3.9%       0.994

######## fast
1924 个 kernel，13 个 step，GPU kernel 时间合计 0.923 ms/step
aten::addmm                                     21.0        22.0       0.206  22.3%       0.546   ← fp16：比 fp32 快 6.9 倍
aten::cudnn_convolution                         12.0        12.0       0.173  18.8%       0.475
aten::copy_                                     28.0        25.0       0.169  18.3%       0.702
```

- **CPU 时间远大于 GPU 时间的算子**（`aten::mul`、`aten::add`）就是 Launch-bound 的来源；反过来，GPU 时间远大于 CPU 时间的算子（`addmm`、`fill_`）才是 GPU 侧要优化的对象。
- `cudnn_convolution` 在慢版中每次调用发起 2.6 个 kernel（31/12），快版是 1 个，多出来的就是 4.2 节提到的布局转换。
- 开销：`--pytorch=autograd-nvtx` 会让快版从 4.4 ms 变成约 5.7 ms/step（`autograd-shapes-nvtx` 约 8 ms），只用于分析，不要用来计时。
- 注意：`record_function` 在这个模式下会额外生成一份带 `op_id` 的同名区间，统计 step 数时要去重（脚本已处理）。

## 4.5 Stream 与 overlap

📎 CUDA 笔记：stream 并不会让单个 kernel 变快，作用是**减少 GPU 空闲**：(1) 拷贝和计算重叠；(2) 占不满 SM 的 kernel 并发执行。[03_streams_overlap.py](03_streams_overlap.py) 在 Thor 上逐一验证：

```
[A1 访存型计算(转 fp16 + GEMM2048)]
只拷贝                               2.374    只计算  1.794
串行（单 stream）                      4.218      1.00x
流水线（双 stream）                     4.140      1.02x
理论下限 max(拷贝, 计算) = 2.374 ms；不重叠时 拷贝+计算 = 4.168 ms

[A2 计算型(GEMM4096)]
只拷贝                               2.380    只计算  7.964
串行（单 stream）                     10.614      1.00x
流水线（双 stream）                    10.675      0.99x
理论下限 max(拷贝, 计算) = 7.964 ms；不重叠时 拷贝+计算 = 10.344 ms
H2D 拷贝带宽 ≈ 105.0 GB/s（pinned → device）

[B1 短 kernel：20 次 softmax(4x256x1024)]
单 stream                          2.356      1.00x
8 个 stream                        3.037      0.78x

[B2 长 kernel：grid=2 的 Triton 循环 kernel]
单 stream                          1.376      1.00x
8 个 stream                        0.206      6.68x
```

**A：拷贝与计算重叠，在 Thor 上没有观察到收益**。用 nsys 查看 A2 的时间线（拷贝在 stream 13，kernel 在 stream 17）：

```
memcpy: 0.000-0.248  1.033-1.374  2.431-2.785  3.546-3.872 ...
kernel: 0.247-1.140  1.373-2.556  2.784-3.650  3.871-5.013 ...
```

拷贝和 kernel **严格交替执行**，每个都在前一个结束之后才开始，尽管它们在不同的 stream 上，而且设备属性显示 `asyncEngineCount = 2`。

> **原因（新容器中用 GPU metrics 查明，见 4.8 B/C）**：在 Thor 上，≥256 KB 的 `cudaMemcpyAsync`（H2D、D2H、D2D 都一样）**不是由拷贝引擎（Copy Engine）执行的，而是由 SM 执行的**。拷贝期间 Copy Engine 的活动为 0，SMs Active 约 99.8%、Warps in Flight 约 71%。拷贝和 GEMM 在抢同一批 SM，所以无法重叠。只有小于约 256 KB 的拷贝才走 Copy Engine，它能和 GEMM 真正并发，但带宽只有约 14~20 GB/s，反而更慢。结论：**在 Thor 上，不要指望通过拷贝/计算重叠来提速**。好在统一内存平台上拷贝本来就很快（0.6 MB 的输入，pinned 拷贝只要几微秒），通常也不是瓶颈。在独显平台上，PCIe 拷贝和计算一般是可以重叠的，这是常用的优化手段。

**B：kernel 并发是有效的，但前提是 kernel 真的占不满 SM，而且运行时间足够长**：
- B2 每个 kernel 只有 2 个 block，单 stream 时 20 个 SM 中有 18 个空着；8 个 stream 让 8 个 kernel 同时运行，快了 6.68 倍。
- B1 的 softmax 每个只要十几微秒，瓶颈在 CPU 侧的 launch 上，多 stream 只会增加切换开销，反而慢了。这种情况要用的是 CUDA Graph 或算子融合。

## 4.6 CUDA Graph

📎 CUDA 笔记：CUDA Graph 把一串 kernel 录制下来，之后一次提交、重复回放，消除每个 kernel 的 launch 开销。**只对 Launch-bound 有效。**

```python
g = torch.cuda.CUDAGraph()
# 1. warmup（在旁路 stream 上进行；cuBLAS/cuDNN 的初始化、选算法不能发生在 capture 期间）
# 2. capture
with torch.cuda.graph(g):
    static_out = model(static_in)
# 3. replay：把新数据拷进静态输入缓冲区，然后回放
static_in.copy_(new_input)
g.replay()          # 结果在 static_out 里
```

### 收益（[04_cuda_graph.py](04_cuda_graph.py)）

```
[ManySmallOps(200)]                 eager 1.667 ms → Graph 0.288 ms   5.79x
[ToyVLA fp16 batch=1]               eager 2.159 ms → Graph 1.092 ms   1.98x
[256MB copy x4 (memory-bound)]      eager 8.917 ms → Graph 8.903 ms   1.00x   ← 无效
```

用 nsys 验证 ToyVLA（`--cuda-graph-trace=node` 展开 graph 内的节点；默认整个 graph 只显示为一个事件）：

```
eager x20:  3.101 ms/iter | launch API/iter 157 | kernels 157/iter, kernel 时间之和 1.103 ms/iter | kernel 间隙中位数 5.1 us
graph x20:  1.129 ms/iter | launch API/iter {'cudaGraphLaunch': 1} | kernels 157/iter, kernel 时间之和 1.101 ms/iter | kernel 间隙中位数 0.0 us
```

GPU 上的 kernel 一个都没少（157 个，总时间都是 1.10 ms），**Graph 只是消除了 kernel 之间的间隙**，于是延迟逼近纯 kernel 时间。256 MB 拷贝那一组本来就只有 4 个很长的 kernel，没有间隙可以消除，所以无效。这正是📎 笔记中"CUDA Graph 对 Memory/Register/Control-flow-bound 无效"的实测验证。

### 限制与常见错误

```
  ✅ 新输入 copy_ 到静态缓冲区后 replay：结果正确 = True
  ⚠️ 没有拷贝新输入就 replay：结果正确 = False（不报错，悄悄用旧输入计算）
  ❌ .item() 同步                     FAIL AcceleratorError CUDA error: operation failed due to a previous error during capture
  ❌ 布尔索引（数据相关 shape）               FAIL ...
  ❌ capture 中创建 CPU 张量并拷到 GPU      FAIL ...
```

| 限制 | 原因 | 解决办法 |
|---|---|---|
| 不能同步（`.item()`、`.cpu()`、`cudaDeviceSynchronize`） | capture 期间 kernel 并不真正执行，没有值可以返回 | 把同步移到 graph 之外 |
| shape 必须固定 | graph 记录的是固定的指针和 launch 配置 | padding/分桶，每种 shape 一个 graph |
| 输入/输出必须是固定的缓冲区 | 同上 | `static_in.copy_(x)`；**忘记拷贝不会报错**，结果会悄悄出错 |
| 不能在 capture 期间做 host 端分配、pageable 拷贝 | 📎 笔记·host-side 内存操作 | 预分配（PyTorch 的 graph 有专用的内存池） |
| 不能有依赖数据的分支 | 只会录下 capture 时走的那条分支 | 改成 `torch.where` 等无分支写法 |

> capture 失败时的错误信息往往很笼统（`operation failed due to a previous error during capture`），并且失败后 CUDA context 会处于异常状态，后续操作全部报错（demo 中把每个失败用例放到子进程里跑，就是这个原因）。排查时可以逐段缩小 capture 的范围。

`torch.compile(mode="reduce-overhead")` 会自动处理静态缓冲区、warmup 和内存池，比手写 CUDA Graph 省心。第 03 章实测 ToyVLA batch=1 的结果是 2.96 倍（比手写 Graph 的 1.98 倍更高，因为还叠加了算子融合）。

## 4.7 GUI 与其他功能（Thor 上的可用性）

### GUI

容器里虽然有 `nsys-ui`，但没有显示器。查看方法：把 `.nsys-rep` 拷贝到有图形界面的电脑上，用对应版本（2025.3）的 Nsight Systems 打开。时间线阅读要点：
- **CUDA HW** 行：GPU 上的 kernel 和拷贝，看空隙、看重叠；
- **CUDA API** 行：CPU 侧的调用，长条的 `cudaStreamSynchronize` / `cudaMemcpy` 就是同步点；
- **NVTX** 行：自己打的标注，对照阶段；
- 选中一个 kernel，按住右键可以看到它对应的 launch 调用（关联箭头）；
- 底部的 Events View / Stats System View 和 `nsys stats` 的内容一样。

### 其他功能在本机的情况（[05_pytorch_python.sh](05_pytorch_python.sh)）

| 功能 | 选项 | Thor 容器 | 说明 |
|---|---|---|---|
| CUDA / NVTX / OSRT / cuBLAS / cuDNN 追踪 | `-t ...` | ✅ | |
| PyTorch 算子自动标注 | `--pytorch=autograd-nvtx` / `autograd-shapes-nvtx` | ✅ | 4.4 |
| CUDA Graph 节点展开 | `--cuda-graph-trace=node` | ✅ | 4.6 |
| 显存使用曲线 | `--cuda-memory-usage=true` | ✅ | |
| **GPU metrics**（GR/SMs Active、SM Issue、Tensor Active、Copy Engine） | `--gpu-metrics-devices=all` | ✅（privileged 容器） | 旧容器报 `Insufficient privilege`。Thor 使用 `gb10b` 指标集，**没有 DRAM 带宽**；见 4.8 |
| **CPU 采样** | `--sample=process-tree`（默认） | ✅（privileged 容器） | 旧容器 `perf_event_open` 失败。数据在 sqlite 的 `COMPOSITE_EVENTS` + `SAMPLING_CALLCHAINS`；慢版 10 步采到 178 个样本，栈顶模块 libc 21%、libcuda 16%、python3.12 13%、libcupti 9%、numpy 8% |
| **Python 采样** | `--python-sampling=true` | ✅（只能在 GUI 中查看调用栈） | 样本以 `NvtxMark` 事件存放（慢版 10 步 183 个，约 1 kHz），调用栈在 NVTX 扩展 payload（schema `Python Periodic Sampling`）里，**sqlite 导出中 `binaryData` 为空**，命令行只能统计样本数。命令行分析 Python 栈继续用 py-spy（第 02 章） |
| **CUDA API 调用栈** | `--cudabacktrace=sync:5000` | ✅（privileged 容器） | `CUDA_CALLCHAINS` 表；最长的一次 `cudaStreamSynchronize`（393 μs）的调用栈直接指向 `at::native::_local_scalar_dense_cuda`，即 `.item()`（慢版的 sync_in_loop） |
| Python 调用栈附在 CUDA API 上 | `--python-backtrace=cuda` | 未在命令行中解析 | payload schema `Python Backtrace on CUDA`，格式同上，需要在 GUI 中查看 |

GPU metrics 正是第 00 章 0.3 节中"空间维度利用率"（SM Active、Tensor Active）的数据来源，实测见 4.8。

## 4.8 GPU metrics：空间维度的利用率、拷贝由谁执行

[07_gpu_metrics.py](07_gpu_metrics.py)，日志 [outputs/07_gpu_metrics.log](outputs/07_gpu_metrics.log)。需要 GPU 计数器权限（privileged 容器或 CAP_SYS_ADMIN），旧容器中不可用。

```bash
nsys profile -t cuda,nvtx --gpu-metrics-devices=all --gpu-metrics-frequency=20000 ...   # 每 50 μs 采样一次
nsys profile --gpu-metrics-set=help        # Thor（GB10B）对应 gb10b，默认自动选择
```

导出 sqlite 后，指标定义在 `TARGET_INFO_GPU_METRICS`，样本在 `GPU_METRICS`（timestamp, metricId, value）。gb10b 提供：GPC/SYS 频率、GR Active、SMs Active、SM Issue、Tensor Active、Compute Warps in Flight、Sync/Async Copy Engine Active。**没有 DRAM 带宽**，原因和 ncu 没有 `dram__*` 指标相同（第 05 章 5.3）。开启 GPU metrics 的额外开销可以忽略（开 / 不开 GPU metrics：慢版 13.96 / 14.16 ms/step，快版 4.47 / 4.35 ms/step，都在运行间波动范围内）。

### A. 流水线各阶段的利用率（呼应第 00 章 0.3 的"空间维度"）

| | GR Active（时间维度） | SMs Active | SM Issue | Tensor Active | Warps in Flight |
|---|---|---|---|---|---|
| 慢版 全部 | 44.1% | 30.8% | 9.8% | 0.6% | 14.7% |
| 慢版 infer | 87.6% | 76.1% | 29.1% | 2.1% | 37.1% |
| 慢版 preprocess / postprocess | 28.1% / 28.5% | 22.7% / 8.5% | 0.0% / 4.4% | 0 / 0.1% | |
| 快版 全部 | 24.6% | 15.6% | 5.0% | 1.6% | 5.7% |
| 快版 infer | 32.9% | 21.2% | 6.7% | 2.3% | 7.6% |

![gpu metrics](outputs/gpu_metrics.png)

读法（对应第 00 章 0.3 的指标体系）：
- **GR Active** 是"时间维度"的忙碌程度，与 4.3 节用 kernel 区间并集算出的 GPU busy 是同一个概念（35.6% / 21.2%），数值偏高一些：50 μs 的采样窗口内只要有活动就会被计入，慢版的 pageable 拷贝尖峰（见 D）也被算作忙。
- **SMs Active / GR Active ≈ 70%（慢）/ 63%（快）**：GPU 忙的时候，平均只有约三分之二的 SM 在干活（第 00 章所说的 Net SM Active），batch=1 的小 kernel 填不满 20 个 SM。
- **SM Issue** 才是"真正在发射指令"的比例：慢版 infer 阶段 SMs Active 76%，Issue 只有 29%；快版只有 5%。**利用率高 ≠ 算力用满**，这一层层下降的数字就是第 00 章那句话的实测版本。
- **Tensor Active** 全程只有 0.6~2.3%：即使快版用了 fp16，ToyVLA 在 batch=1 时的矩阵太小，Tensor Core 几乎一直空闲。要提高它，需要的是更大的 batch 或融合，而不是更快的 GEMM kernel（第 05 章 5.9 有同样的结论）。
- 图中慢版第二步 preprocess 里有一段约 2.5 ms、SMs Active 75% 但 SM Issue 为 0 的平台：SM 被占着却不发射指令，这是 pageable H2D 拷贝在等主机侧的中转缓冲，并且只在 nsys 下出现（D）。

### B. 拷贝与 GEMM 为什么不重叠（4.5 节的现象）

8 个 32 MB 的 pinned H2D 拷贝（copy stream）+ 8 次 fp16 GEMM 4096（compute stream），重复 3 次：

| | 拷贝期间 | GEMM 期间 |
|---|---|---|
| SMs Active | **99.8%** | 99.1% |
| Compute Warps in Flight | **72.7%** | 17.1% |
| SM Issue | 7.5% | 3.1% |
| Tensor Active | 8.5% | 69.2% |
| **Copy Engine 0 / 1** | **0 / 0** | 0 / 0 |

拷贝期间 **Copy Engine 完全没有活动，SM 却全部被占满**：大块 `cudaMemcpyAsync` 是驱动用 SM 执行的（类似一个拷贝 kernel）。拷贝和 GEMM 抢的是同一批 SM，所以只能交替执行（memcpy 合计 8.35 ms，GEMM 23.80 ms，区间只在交界处重叠了 2.40 ms，墙钟 30.26 ms，比两者之和只少约 6%）。D2D 拷贝的情况完全相同（SMs Active 99.8%，CE 为 0）。

### C. 拷贝由谁执行，与拷贝大小有关

单独做 pinned H2D 拷贝（每种大小合计 ≥ 8 MB），统计拷贝进行期间的样本：

| 每次拷贝 | 每次耗时 | 带宽 | 拷贝期间 SMs Active | Copy Engine 0 |
|---|---|---|---|---|
| 64 KB | 3.4 μs | 19.4 GB/s | 0 | **58.7%** |
| 256 KB | 3.5 μs | 74.5 GB/s | 44.3% | 0 |
| 512 KB | 5.5 μs | 94.7 GB/s | 57.5% | 0 |
| 1 MB | 6.0 μs | 175.9 GB/s | （没有样本落在拷贝区间内） | |
| 4 MB | 20.2 μs | 207.7 GB/s | 78.0% | 0 |
| 16 MB | 156.1 μs | 107.5 GB/s | 95.0% | 0 |
| 32 MB | 317.6 μs | 105.6 GB/s | 98.8% | 0 |

- **< 256 KB 的拷贝走 Copy Engine，≥ 256 KB 的拷贝由 SM 执行**（阈值在 64 KB 和 256 KB 之间，未进一步细分）。
- 1~4 MB 的带宽超过了 DRAM 峰值，因为同一块数据被反复拷贝，源数据留在了 32 MB 的 L2 里；16 MB 以上才是真实的内存带宽（约 106 GB/s，读写各一次，合计约 212 GB/s）。1 MB 那一行每次只有 6 μs，比 50 μs 的采样间隔短得多，没有样本落进去，这是采样类工具的固有限制。

和 8 次 GEMM 并发（拷贝总量都是 128 MB）：

| 分块大小 | 执行单元 | GEMM + 拷贝（各自单独的耗时之和） | 区间重叠 | 墙钟 |
|---|---|---|---|---|
| 64 KB × 2048 | Copy Engine（拷贝期间 CE0 87%、Tensor Active 49%） | 7.72 + 9.52 = 17.24 ms | **5.56 ms** | 15.95 ms |
| 256 KB × 512 | SM | 7.71 + 2.68 = 10.39 ms | 0.64 ms | 11.15 ms |
| 16 MB × 8 | SM | 7.57 + 1.63 = 9.19 ms | 0.94 ms | **8.39 ms** |

走 Copy Engine 的小拷贝**确实能和 GEMM 并发**（重叠 5.56 ms），但 Copy Engine 的带宽只有约 14 GB/s，128 MB 要 9.5 ms，总时间反而是用 SM 拷贝大块的近 2 倍。**在 Thor 上，大块拷贝 + 不追求重叠是最快的**。更好的做法是从源头上减少拷贝：Thor 的 CPU 和 GPU 共享同一块 DRAM，可以让 GPU 直接读取 pinned 内存（zero-copy），或者把预处理直接放到 GPU 上（未实测，留给第 07 章）。

### D. 观察者效应：nsys 下的 pageable 拷贝尖峰

| 报告 | H2D 拷贝 | 中位耗时 | 超过 1 ms 的次数 | 最长 |
|---|---|---|---|---|
| 慢版，nsys（01_nsys_basics.sh，不开 GPU metrics） | 33 次，pageable | 10.7 μs | **9** | 4187 μs |
| 慢版，nsys + GPU metrics | 23 次，pageable | 9.7 μs | **8** | 6007 μs |
| 快版，nsys | 33 次，pinned | 5.1 μs | 0 | 14 μs |
| 快版，nsys + GPU metrics | 23 次，pinned | 6.2 μs | 0 | 13 μs |

**不开 nsys** 直接给慢版的 preprocess 计时（60 步）：中位 3.09 ms，最长 3.31 ms，没有一次超过 5 ms。所以这些 1~6 ms 的尖峰是 **nsys 跟踪 pageable 拷贝时引入的**，在真实运行中不存在，也说明 T01 中"no_pin 对延迟几乎没有影响"的结论在不开 nsys 时仍然成立。教训：在 profiler 里看到的异常，要在不开 profiler 的情况下复现一次，再下结论。

## 常见坑

1. **`--capture-range-end` 默认是 `stop-shutdown`**：遇到 `cudaProfilerStop` 后 nsys 会直接结束程序，后面的代码都不执行。
2. **计时包含了 `profiler.stop()`**：nsys 落盘需要约 1 s（4.1 节实测踩坑）。
3. **采集了启动和 warmup 阶段**：用 `--capture-range=cudaProfilerApi`，或者用 `--delay` / `--duration`。
4. **`gpu_gaps` 默认阈值是 500 ms**：推理场景要调到 0.1~1 ms。
5. **默认看不到 CUDA Graph 内部的 kernel**：加上 `--cuda-graph-trace=node`。
6. **sqlite 是旧的**：`nsys stats` 默认会复用已有的 sqlite，可能导致"File is older than input file"，加上 `--force-export=true`。
7. **以为 stream 一定能带来重叠**：在 Thor 上，拷贝和计算实测没有重叠；短 kernel 用多 stream 反而更慢（4.5）。
8. **把 CUDA Graph 用在 Memory-bound 负载上**：实测 1.00 倍，完全无效（4.6）。
9. **在 nsys 中看到 pageable 拷贝偶尔要几毫秒，就认定是程序的问题**：这是 nsys 的观察者效应，不开 nsys 时不存在（4.8 D）。
10. **以为 `cudaMemcpyAsync` 一定由拷贝引擎执行**：Thor 上 ≥256 KB 的拷贝由 SM 执行，会和 kernel 抢 SM（4.8 B/C）。
11. **随手分享 `.nsys-rep` / `.sqlite`**：nsys 会把目标进程的**完整环境变量**写进报告（sqlite 中的 `TARGET_INFO_SYSTEM_ENV` 表），如果环境里有云服务密钥、token，它们也会被一起写进去。分享报告之前先检查，或者用干净的环境（`env -i PATH=... python ...`）采集。

## 练习

见 [exercises.md](exercises.md)。

## 小结

- nsys 是 Thor 上**唯一能拿到 GPU 时间**的工具，开销约 +30%。
- 方法：只采集稳态 + NVTX 标注阶段 → `stats` / `analyze` 看总体 → SQL 做定制分析。
- 慢流水线的诊断结论全部在时间线上得到了印证：预处理 CPU-bound、推理的 GPU 工作溢出到后处理、后处理 Launch-bound；快版仍然是 Launch-bound（GPU busy 21%）。
- stream：Thor 上拷贝和计算实测不重叠，原因是 ≥256 KB 的拷贝由 SM 执行（GPU metrics 实测，4.8）；kernel 并发只对"长而窄"的 kernel 有效。
- GPU metrics（4.8）：GR Active → SMs Active → SM Issue 逐层下降（慢版 infer：88% → 76% → 29%），Tensor Active 只有约 2%，这是"利用率高 ≠ 算力用满"的实测版本。CUDA Graph：只对 Launch-bound 有效（5.79x / 1.98x / 1.00x）。
- 下一步：nsys 告诉我们"GPU 在干什么"，要知道"某个 kernel 为什么慢"，就要用[第 05 章 Nsight Compute](../05_nsight_compute/README.md)。
