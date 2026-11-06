# 模型性能分析与调优教程

> 目标：建立"**先定位瓶颈，再针对性优化**"的方法论，掌握从系统级 → 框架级 → Kernel 级的完整工具链。
> 运行平台：Jetson AGX Thor（JetPack 7.2 / L4T 38.4，aarch64，MAXN + jetson_clocks），同时兼顾 x86 + 独立 GPU 的差异。
> Python 环境：`/opt/venv/bin/python`（torch 2.9.1）；额外依赖见 [requirements.txt](requirements.txt)，环境细节与平台限制见 [plan/env_report.md](plan/env_report.md)。
> 已有知识沉淀：[doc/](doc/) 下的 xmind 笔记，各章标注了 **📎 对应笔记**。
> 开发过程（面向规格）：[plan/spec.md](plan/spec.md) → [plan/plan.md](plan/plan.md) → [plan/tasks.md](plan/tasks.md)（含每章的实测记录）。
> 运行环境：新容器 v1.3.0_thor（`--privileged`）。ncu、nsys GPU metrics、CPU 采样需要 privileged（或 CAP_SYS_ADMIN / PERFMON / SYS_PTRACE），详见 env_report。
> 产物：各章 `outputs/` 只保留日志、图和小文件。`.nsys-rep` / `.sqlite` / `.ncu-rep` / `.engine` / `.onnx` 和大于 5 MB 的 trace 已在收尾时删除（体积大；**nsys 报告包含采集时的完整环境变量**），重新运行对应章节的脚本即可生成。第 08 章的 S8/S9 依赖第 06 章生成的 engine：先运行 `06_tensorrt/01_export_onnx.py` 和 `02_trtexec.py`。

## 教程目录

| 章 | 主题 | 状态 |
|---|---|---|
| [common/](common/) | 共享组件：benchmark、示例模型、慢流水线、监控采样器、CUPTI 修复 | ✅ |
| [00_theory](00_theory/README.md) | 基础理论：利用率指标、Roofline（实测）、瓶颈分类、正确计时 | ✅ |
| [01_system_monitor](01_system_monitor/README.md) | 系统级监控：jtop / tegrastats / nvidia-smi，5 种负载的系统特征 | ✅ |
| [02_python_profilers](02_python_profilers/README.md) | Python 级：cProfile / line_profiler / py-spy / VizTracer | ✅ |
| [03_torch_profiler](03_torch_profiler/README.md) | torch.profiler：算子、同步检测、显存、compile 诊断（⚠️ Thor 上没有 GPU 时间） | ✅ |
| [04_nsight_systems](04_nsight_systems/README.md) | Nsight Systems：时间线、SQL 分析、stream、CUDA Graph、算子级 GPU 时间、GPU metrics | ✅ |
| [05_nsight_compute](05_nsight_compute/README.md) | Nsight Compute：过滤与开销、Thor 上的 DRAM 换算、四类瓶颈 kernel 的修复前后对比、Tensor Core、Roofline、流水线真实 kernel | ✅ |
| [06_tensorrt](06_tensorrt/README.md) | TensorRT：trtexec、逐层 profile + Engine Inspector、三步法（TRT Profiler → nsys → ncu）、PyTorch vs TensorRT | ✅ |
| [07_optimization](07_optimization/README.md) | 常见优化手段：瓶颈 → 手段 → 实测收益总表、无效/负优化案例、prefill vs decode | ✅ |
| [08_case_study](08_case_study/README.md) | 综合实战：漏斗诊断 → 9 步修复（10.3 → 0.75 ms，13.8x）、报告模板与实测报告 | ✅ |
| [appendix](appendix/README.md) | 附录：命令速查、工具对比（含实测开销）、可视化流程、38 条常见坑、参考资料 | ✅ |

> 以下为原始大纲（各章均已展开，保留作为设计记录）。

---

## 总体分析路径（漏斗模型）

```
 ┌──────────────────────────────────────────────────────┐
 │ L0 系统级监控   jtop / tegrastats / nvidia-smi dmon   │  GPU 忙不忙？功耗/温度/频率/EMC 带宽？
 ├──────────────────────────────────────────────────────┤
 │ L1 Python 级    cProfile / py-spy / VizTracer        │  CPU 侧时间花在哪？预处理/后处理/调度？
 ├──────────────────────────────────────────────────────┤
 │ L2 框架/引擎级  torch.profiler / TRT Profiler        │  "哪个算子/哪一层慢？"
 ├──────────────────────────────────────────────────────┤
 │ L3 系统时间线   Nsight Systems (nsys)                │  "为什么慢？" GPU 空隙/同步/拷贝/overlap
 ├──────────────────────────────────────────────────────┤
 │ L4 Kernel 级    Nsight Compute (ncu)                 │  "这个 kernel 为什么慢？" 算力/带宽/占用率
 └──────────────────────────────────────────────────────┘
        自顶向下逐层缩小范围，避免一上来就用 ncu
```

**一句话分工**（📎 TensorRT 笔记·区别联系）：TRT/torch Profiler 回答"哪层慢"，Nsight Systems 回答"为什么慢"，Nsight Compute 回答"这个 kernel 为什么慢"。框架级工具懂模型语义但看不到硬件行为；Nsight 能看到硬件行为，但只认 `sm80_xmma_128x64...` 这类 kernel 名，不知道它对应哪一层。所以要**配合使用**，用 NVTX 把两边对上。

---

## 第 0 章 基础理论：性能瓶颈与度量

- 0.1 为什么要先 profile：Amdahl 定律、"过早优化是万恶之源"
- 0.2 GPU 执行模型速览（📎 CUDA 笔记·硬件架构，作为前置知识引用）
  - CPU 与 GPU 的协作：GPU 是 CPU 的协处理器、kernel launch 队列、异步执行
  - Grid / Block / Warp / Thread、SIMT、SM
  - 内存层级：Register → Shared Memory → L1/L2 → Global(DRAM)，以及各级的容量和延迟
  - 理论算力 = 核数 × 主频 × 每周期操作数；带宽 = 位宽 × 频率 × prefetch
  - PCIe / NVLink / 统一内存 / 零拷贝；**Jetson 的 CPU 和 GPU 共享物理 DRAM**，对 H2D 拷贝和带宽争用有什么影响
- 0.3 **GPU"利用率"指标体系**（📎 CUDA 笔记·性能分析·第一部分）

  | 层次 | 指标 | 回答的问题 | 来源工具 |
  |---|---|---|---|
  | 功耗 | Power | 是否真的在"出力" | jtop / nvidia-smi |
  | 时间维度 | GPU Util（`dmon` 的 `sm` 列） | 有多少时间有 kernel 在跑 | nvidia-smi dmon / jtop |
  | 空间维度 | SM Active / SM Efficiency | 忙的时候多少 SM 在干活 | nsys `--gpu-metrics-devices` / DCGM |
  | Warp 维度 | Achieved Occupancy | 一个 SM 内实际驻留了多少 warp | ncu |
  | Tensor Core | Tensor Active | 矩阵乘有没有走 Tensor Core | nsys GPU metrics / ncu |
  | 带宽 | `dmon` 的 `mem` 列 | 显存控制器**忙碌时间占比**（不是容量占用，也不是 GB/s 占峰值） | nvidia-smi / ncu DRAM Throughput |
  | 有效产出 | **MFU / HFU** | 实际有效 FLOPs ÷ 理论峰值 | 根据模型 FLOPs 手算 |

  - SM Active 的三种聚合方式：Gross / Net / Effective
  - 核心认知：上表除 MFU/HFU 外，其余都只说明硬件"忙不忙"，不说明"产出多少"。利用率高 ≠ 算力用满；利用率低则一定没用满
  - MFU 计算：Transformer 训练 ≈ 6·N·tokens/s ÷ 峰值，推理 ≈ 2·N·tokens/s ÷ 峰值；CNN 按逐层 FLOPs 计算；HFU ≥ MFU，差值就是重计算这类额外开销
- 0.4 **Roofline 模型**（📎 CUDA 笔记·Roofline model）
  - 屋顶 = 峰值算力，房檐斜率 = 峰值带宽，拐点 = 最佳计算强度
  - 怎么确定模型/算子在图上的位置：横轴 = 理论 FLOPs ÷ 理论访存字节，纵轴 = 实测 FLOPs/s
  - 两种优化在图上的移动：**算子融合**减少访存，点向右移；**实现优化**（合并访存、Tensor Core）点向上移
  - 案例：VGG16 vs MobileNet；扩展到 LLM 推理的 prefill（偏算力瓶颈）vs decode（偏带宽瓶颈）
  - 实践：算出 Thor 的峰值算力/带宽，画出 Roofline，并标出几个典型算子
- 0.5 **四大 GPU 瓶颈类型**（📎 CUDA 笔记·第二部分），另加 CPU 侧瓶颈

  | 类型 | 本质 | 诊断特征 | 验证工具 | 典型修复 |
  |---|---|---|---|---|
  | CPU / Python bound | Python 开销、预处理 | GPU 时间利用率低，CPU 单核跑满 | jtop、VizTracer | 向量化、多进程、GPU 预处理 |
  | 同步 bound | `.item()`、`cudaDeviceSynchronize` | 时间线上出现 CPU 阻塞等待 | torch.profiler、nsys | 去掉同步、异步拷贝 |
  | Launch-bound | kernel 启动往返延迟 | GPU Busy < 50% 且 kernel 很多很碎 | nsys 看 kernel 间的 gap | CUDA Graph、kernel 融合 |
  | Memory-bound | 带宽打满 | Memory BW > 70% Peak，SM 利用率低 | ncu 看 Memory 吞吐 | 融合、低精度、数据排布 |
  | Register-bound | 寄存器压力大导致占用率低 | Occupancy < 40%，Regs/Thread > 64 | ncu 看 Occupancy | 精简变量、`-maxrregcount`、调 block size |
  | Control-flow-bound | warp divergence | IPC < 1.5 | ncu 看 IPC / Branch | 减少分支、改用无分支写法 |

  - 关键结论：**CUDA Graph 只对 Launch-bound 有效**，对其余几类无效 → 先诊断，再选手段
- 0.6 正确的计时方法
  - CUDA 是异步执行的 → 用 `torch.cuda.synchronize()` / `torch.cuda.Event` 计时（📎 CUDA 笔记·STREAM 和 EVENT）
  - 先 warmup，再多次测量；报告 P50/P90/P99；区分 latency 和 throughput
  - Jetson：测 benchmark 前锁频（`jetson_clocks`、`nvpmodel`），注意温度降频
- 实践：写一个通用 benchmark 工具 `bench_utils.py`（含 warmup、Event 计时、统计分位数、记录频率/功耗）

## 第 1 章 系统级监控：jtop / tegrastats / nvidia-smi

- 1.1 jetson-stats 简介与安装（📎 性能分析工具笔记·jtop，含各页签截图和字段说明）
- 1.2 各页签解读
  - ALL：Model、MEM/Swap、EMC、NV Power、Jetson Clocks、进程表（PID/GPU MEM/类型 I/G 等）
  - GPU：GPU 利用率、频率、调频策略（gov：`nvhost_podgov`）
  - CPU / MEM / ENG（HW engines）：APE、PVA、DLA、NVENC/NVDEC、NVJPG、VIC、SE
  - CTRL：nvpmodel 功耗模式、jetson_clocks、风扇 Profile
  - INFO：JetPack / L4T / CUDA / cuDNN / TensorRT 版本
- 1.3 判读重点：GPU% 与 EMC% 一起看（Jetson 上 EMC 是 CPU 和 GPU 共享的带宽）；看温度和功耗是否触发降频
- 1.4 jtop Python API：推理脚本运行时采集指标，然后与延迟曲线对齐
- 1.5 tegrastats 命令行采集 + 日志解析绘图（Thor 上需确认路径和是否可用）
- 1.6 nvidia-smi
  - `nvidia-smi dmon -s u`：按时间线统计 sm/mem/enc/dec 利用率（📎 已验证 Thor 可用）
  - `mem` 列的常见误解（见 0.3）
  - NVML / `pynvml` Python 接口
  - x86 对照：`nvtop`、DCGM（`dcgmi dmon`，字段 `SM_ACTIVE`/`SM_OCCUPANCY`/`TENSOR_ACTIVE`，可用于生产环境的低开销监控）
- 实践：运行一个模型推理，同时观察 jtop 和 dmon，初步判断是 CPU 还是 GPU 受限；对比不同 nvpmodel 模式下的延迟

## 第 2 章 Python 级 Profiling：cProfile / py-spy / VizTracer

- 2.1 cProfile + snakeviz：函数级统计（入门、零依赖）
- 2.2 line_profiler：逐行耗时
- 2.3 py-spy：采样式、无侵入、可以 attach 到运行中的进程，输出火焰图
- 2.4 **VizTracer**（📎 性能分析工具笔记·viztracer）
  - 原理：确定性 tracing，记录每次函数调用的时间线
  - 用法：命令行 `viztracer xxx.py`、代码内 `with VizTracer()`；用 `vizviewer` 或 **Perfetto UI**（ui.perfetto.dev）查看
  - 导出火焰图（`vizviewer --flamegraph`）
  - 控制开销：`max_stack_depth`、`include_files`/`exclude_files`、`min_duration`、`log_sparse`
  - 自定义事件：`log_var`、`instant event`、`@trace_and_save`
  - 多进程 / 多线程 / async 支持（DataLoader workers、ROS 节点等）
- 2.5 工具对比：开销、精度、适用场景
- 2.6 局限：只能看到 CPU 侧。由于 CUDA 异步执行，Python 层看到的耗时不等于 GPU 实际执行时间
- 实践：分析 "读图 → 预处理 → 推理 → 后处理" 流水线，找出 CPU 侧热点

## 第 3 章 框架级 Profiling：torch.profiler

- 3.1 基础 API：`activities=[CPU, CUDA]`、`record_shapes`、`profile_memory`、`with_stack`、`with_flops`
- 3.2 `key_averages().table()`：按 `self_cuda_time_total` 排序、`group_by_input_shape`
- 3.3 `record_function` 自定义标注
- 3.4 `schedule(wait, warmup, active, repeat)` 与 `on_trace_ready`
- 3.5 导出 Chrome trace，用 Perfetto 查看时间线（CPU op ↔ CUDA kernel 的关联箭头）
- 3.6 TensorBoard 插件 / **HTA（Holistic Trace Analysis）**：GPU idle 分解、kernel 分类、计算与通信/拷贝的重叠
- 3.7 显存分析
  - `memory_summary`、`max_memory_allocated`、Memory Snapshot（`_record_memory_history` + memory_viz）
  - **Caching allocator 与显存池**（📎 CUDA 笔记·显存池与生命周期管理）：为什么不应每次 `cudaMalloc/free`，碎片成因，`PYTORCH_CUDA_ALLOC_CONF`
- 3.8 常见问题模式识别
  - 同步点（`.item()`、`print(tensor)`、`nonzero`、布尔索引）
  - 小 kernel 太多 → 属于 Launch-bound
  - H2D/D2H 拷贝、`pin_memory`/`non_blocking`（Jetson 统一内存下的差异）
  - dtype 隐式转换、非连续 tensor 导致额外的 copy kernel
- 3.9 `torch.compile`：`TORCH_LOGS`、graph break 分析、`mode="reduce-overhead"`（即 CUDA Graph）
- 实践：对 VLA / Transformer 模型做推理 profile，输出 Top-N 耗时算子并给出瓶颈分类

## 第 4 章 系统时间线：Nsight Systems (nsys)

- 4.1 定位：全系统时间线，覆盖 CPU 线程、CUDA API、Kernel、Memcpy、OS runtime、cuDNN/cuBLAS/TensorRT。它不关心你在跑哪一层，只关心 GPU 实际在干什么
- 4.2 命令行采集
  - `nsys profile -t cuda,nvtx,osrt,cudnn,cublas,tensorrt -o report python xxx.py`
  - 只抓关键区间：`--capture-range=cudaProfilerApi|nvtx`、`--delay`、`--duration`
  - `--cuda-memory-usage=true`、`--sample=cpu`、`--cudabacktrace`
  - **GPU Metrics**：`--gpu-metrics-devices=help` 查看 Thor 是否支持 → SM Active、Tensor Active、DRAM 带宽随时间变化的曲线
- 4.3 **NVTX 标注**：`torch.cuda.nvtx.range_push/pop`、`emit_nvtx()`、`nvtx` 包。这是"kernel ↔ 模型层"对应的关键
- 4.4 GUI 分析（在主机上用 nsys-ui 打开从 Jetson 拷回的 `.nsys-rep`）
  - 时间线阅读：kernel gap（GPU bubble）、CPU launch 延迟、多 stream 并发
  - Events View / Stats System View
- 4.5 命令行统计：`nsys stats`（cuda_gpu_kern_sum、cuda_api_sum、nvtx_sum）、`nsys analyze` 专家规则、导出 sqlite 做自定义分析
- 4.6 **Stream / Event 与 overlap**（📎 CUDA 笔记·STREAM 和 EVENT）
  - 默认流的串行和阻塞；多 stream 可以让拷贝与计算重叠，SM 没占满时 kernel 之间也能重叠
  - stream 不会让单个 kernel 变快，作用是减少 GPU 空闲时间
  - 在时间线上识别没有 overlap 的地方
- 4.7 **CUDA Graph 诊断与限制**（📎 CUDA 笔记·host-side 内存操作 / 显存分配方式）
  - 时间线上 Graph 前后的对比（gap 消失）
  - 与 Graph 不兼容的情况：host 端动态 `cudaMalloc`、device 端 `malloc`、同步点、依赖数据的分支、动态 shape → 解决办法是预分配 + 固定 shape
- 4.8 典型案例：GPU 等 CPU / 拷贝串行 / 同步点 / TRT 中大量 reformat kernel
- 实践：用 nsys + NVTX 分析一次完整推理（预处理 → 模型 → 后处理），量化 GPU bubble 占比

## 第 5 章 Kernel 级：Nsight Compute (ncu)

- 5.1 定位：单个 kernel 的深度硬件分析。replay 机制开销大，只用于离线分析
- 5.2 命令行采集
  - `ncu --set full -o report python xxx.py`
  - 过滤：`-k regex:gemm`、`--launch-skip`、`--launch-count`、`--nvtx --nvtx-include`
  - 权限问题（`ERR_NVGPUCTRPERM`）、Jetson 上的配置、锁频（`--clock-control`）
- 5.3 报告解读，按 Section 对应瓶颈类型
  - **SOL**：Compute vs Memory 吞吐占峰值的比例 → 初步判断瓶颈类型
  - **Roofline** 图（与 0.4 呼应）
  - **Occupancy / Launch Stats**：寄存器、shared memory、block size 三种限制 → 对应 Register-bound
  - **Memory Workload**：L1/L2 命中率、DRAM 带宽、**合并访存**、**shared memory bank conflict**（📎 CUDA 笔记·bank）
  - **Warp State / Stall Reasons**：Long Scoreboard、Barrier、MIO Throttle、Math Pipe Throttle 等
  - **Compute Workload / IPC / Branch**：warp divergence → 对应 Control-flow-bound（📎 CUDA 笔记·Warp divergence）
  - **Tensor Core 利用率**：确认 GEMM 走了 Tensor Core 路径（dtype 和维度对齐）
  - Source 页：SASS/PTX ↔ 源码热点
- 5.4 Baseline 对比：同一 kernel 优化前后做 diff
- 5.5 联动 `../triton`、自定义 CUDA kernel（shuffle 归约、shared memory tiling 的效果验证）
- 实践：分别构造 Memory-bound（elementwise）、Register-bound、Control-flow-bound、bank conflict 四个 kernel，用 ncu 逐一识别并修复

## 第 6 章 推理引擎：TensorRT 性能分析（📎 TensorRT 笔记·性能分析）

- 6.1 trtexec 基准测试
  - 构建：`--onnx --saveEngine --fp16/--int8/--fp8 --memPoolSize`
  - 性能：延迟/吞吐、`--useCudaGraph`、`--noDataTransfers`、`--useSpinWait`
- 6.2 **TRT 内置 Profiler**：`--dumpProfile --separateProfileRun --exportProfile=profile.json --profilingVerbosity=detailed --dumpLayerInfo --exportLayerInfo`；`IProfiler` 接口；Engine Inspector
- 6.3 **三步法：TRT Profiler → nsys → ncu**
  1. TRT Profiler 找到最慢的层（如 `MatMul_387 40ms`）
  2. `nsys profile -t cuda,nvtx,tensorrt trtexec ...` 分析这一层为什么慢：Tensor Core 没启用 / reformat（NCHW↔NHWC）/ kernel 太碎 / 没有 overlap
  3. ncu 深挖这一层对应的 kernel：occupancy、带宽、warp stall、Tensor Core 利用率
  - 如何把 layer 和 kernel 对应起来：`profilingVerbosity=detailed` 时 TRT 会输出 NVTX
- 6.4 精度与性能的权衡：FP16/INT8/FP8、混合精度、Polygraphy 排查精度问题
- 6.5 Plugin 性能：预分配 workspace，兼容 CUDA Graph（呼应 4.7）
- 6.6 torch2trt / ONNX Runtime profiling（联动 `../torch2trt`）
- 6.7 Jetson 上的 DLA / PVA：能否 offload，在 jtop ENG 页确认是否在用
- 实践：同一模型 PyTorch eager vs torch.compile vs TensorRT FP16/INT8 的对比，并走一遍三步法

## 第 7 章 常见优化手段（与瓶颈一一对应）

| 瓶颈 | 诊断工具 | 优化手段 |
|---|---|---|
| Python/CPU 开销 | VizTracer、py-spy | 向量化、减少 Python 循环、多进程预处理、C++ 扩展、CUDA 预处理（📎 CUDA 笔记·使用 CUDA 进行预处理/后处理） |
| 数据加载 | torch.profiler、nsys | `num_workers`、`pin_memory`、预取、DALI、GPU 解码（NVJPG/NVDEC） |
| 同步点 | torch.profiler、nsys | 去掉 `.item()`、异步拷贝、Event 跨流同步 |
| Launch-bound | nsys | CUDA Graph、`torch.compile(mode="reduce-overhead")`、算子融合 |
| 拷贝/计算没有重叠 | nsys | 多 stream、pinned memory、双缓冲 |
| Memory-bound | ncu | 算子融合、低精度、减少中间张量、FlashAttention、优化数据排布 |
| Compute-bound | ncu | Tensor Core（维度对齐）、TF32/FP16/FP8、cuBLASLt/CUTLASS/Triton |
| Register-bound | ncu | 精简局部变量、`-maxrregcount`、调整 block size |
| Control-flow-bound | ncu | 减少分支、按 warp 对齐数据、无分支写法 |
| 显存分配/碎片 | memory snapshot、nsys | 预分配、显存池（`cudaMallocAsync`/mempool）、allocator 配置 |
| 动态 shape | nsys、compile 日志 | shape 分桶、padding、TRT optimization profile |

- 7.1 LLM / VLA 推理专项：prefill vs decode、KV cache、batch、投机解码、量化（AWQ/GPTQ/FP8）
- 7.2 优化闭环：度量 → 定位 → 假设 → 修改 → 验证（避免负优化），每一步都要记录基线

## 第 8 章 综合实战

- 8.1 案例：VLA 模型（联动 `../vla`）端到端推理延迟优化
  1. jtop + dmon 看整体资源，锁频后建立基线
  2. VizTracer 看 CPU 侧流水线
  3. torch.profiler 找 Top 算子
  4. nsys 看时间线：GPU bubble、同步、overlap
  5. ncu 深挖热点 kernel，按四类瓶颈归类
  6. 导出 TensorRT 并走三步法
  7. 逐项优化，记录每一步的收益（延迟、MFU、功耗）
- 8.2 输出性能分析报告模板 `report_template.md`

## 附录

- A. 工具速查表（命令 + 常用参数，一页纸）
- B. 工具对比：开销 / 粒度 / 能否看到 GPU / 是否侵入 / 是否懂模型语义 / 查看方式
- C. 可视化：Perfetto UI、TensorBoard、nsys-ui、ncu-ui（Jetson 采集、主机查看的流程）
- D. 常见坑：没有 warmup、计时前没同步、profiler 自身开销、Jetson 没锁频或温度降频、计数器权限、把 `dmon` 的 mem 列当成显存占用、只看 GPU 利用率
- E. 参考资料：PyTorch Profiler、HTA、VizTracer、Nsight Systems/Compute User Guide、TensorRT Best Practices、jetson-stats，以及 doc/ 脑图里收录的链接

---

## 建议目录结构

```
profile/
├── README.md                 # 本大纲
├── doc/                      # 已有脑图笔记（xmind）
├── 00_theory/                # 利用率指标、Roofline、四大瓶颈、计时方法、bench_utils.py
├── 01_system_monitor/        # jtop / tegrastats / nvidia-smi
├── 02_python_profilers/      # cProfile / py-spy / viztracer
├── 03_torch_profiler/
├── 04_nsight_systems/        # 含 stream / CUDA Graph 案例
├── 05_nsight_compute/        # 含四类瓶颈 kernel 案例
├── 06_tensorrt/              # trtexec / TRT Profiler / 三步法
├── 07_optimization/
├── 08_case_study/
└── appendix/
```

每章包含：`README.md`（原理与讲解）、`xx_demo.py`（可运行示例）、`exercises.md`（练习与思考题）。
