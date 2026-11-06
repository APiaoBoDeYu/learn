# 第 05 章 Kernel 级：Nsight Compute（ncu）

> 📎 对应笔记：[doc/CUDA.xmind](../doc/CUDA.xmind) →「性能分析·第二部分：四大 GPU 瓶颈类型」「硬件架构·bank」「硬件架构·Warp divergence」「CUDA 寄存器、共享内存」；[doc/TensorRT_副本.xmind](../doc/TensorRT_副本.xmind) →「三步法」第三步
>
> 漏斗模型的最底层（L4）：nsys 告诉我们"GPU 时间花在哪些 kernel 上"，ncu 回答"**这个 kernel 为什么慢**"。它通过反复重放（replay）同一个 kernel、每次读取一组硬件计数器，给出几千个指标，以及规则引擎的优化建议。
>
> 本章所有数字都在新容器（privileged，v1.3.0_thor）中实测。旧容器没有 CAP_SYS_ADMIN，ncu 报 `ERR_NVGPUCTRPERM`，见 [plan/env_report.md](../plan/env_report.md)。

## 学习目标

1. 会用过滤选项（`-k`、`--launch-skip/-c`、`--nvtx-include`、`--profile-from-start off`）只采集需要的 kernel，理解 replay 带来的开销；
2. 知道 **Thor（统一内存）上 ncu 的特殊之处**：没有 DRAM 指标、SOL 的 Memory% 有误导、锁频不生效，以及如何自己换算 DRAM 带宽；
3. 能从 ncu 的指标中识别四类 kernel 级瓶颈：Memory-bound、Register-bound、Control-flow-bound、shared memory bank conflict，并用前后对比验证修复；
4. 会用 Source 页把问题定位到源码行，会用 Tensor Core 指标确认 GEMM 走了哪条路径；
5. 对笔记里的经验阈值（"Occupancy < 40%""IPC < 1.5"）保持怀疑：**它们只是线索，判断依据是实测**。

## 目录

- [5.1 定位与工作原理](#51-定位与工作原理)
- [5.2 命令行采集](#52-命令行采集)
- [5.3 Thor 上的特殊之处](#53-thor-上的特殊之处)
- [5.4 Memory-bound：合并访存、向量化、融合](#54-memory-bound合并访存向量化融合)
- [5.5 Register-bound：Occupancy 低到底是不是问题](#55-register-boundoccupancy-低到底是不是问题)
- [5.6 Control-flow-bound：warp divergence](#56-control-flow-boundwarp-divergence)
- [5.7 Shared memory bank conflict](#57-shared-memory-bank-conflict)
- [5.8 Tensor Core 与 Roofline](#58-tensor-core-与-roofline)
- [5.9 分析流水线中的真实 kernel](#59-分析流水线中的真实-kernel)
- [5.10 按 Section 解读报告：速查](#510-按-section-解读报告速查)
- [常见坑](#常见坑) · [练习](exercises.md) · [小结](#小结)

| 文件 | 内容 |
|---|---|
| [cuda/bottlenecks.cu](cuda/bottlenecks.cu) | 本章所有自定义 CUDA kernel（每组都有"问题版"和"修复版"） |
| [kernels.py](kernels.py) | 用 nvcc 编译成 `.so`、ctypes 调用；`resource_table()` 解析 `--resource-usage` |
| [ncu_tools.py](ncu_tools.py) | 脚本"自己 profile 自己"：计时 → 在 ncu 下以 `--worker` 运行自身 → 用 `ncu_report` 读报告出对比表；Thor 的 DRAM 换算 |
| [01_ncu_basics.sh](01_ncu_basics.sh) + [workload_basic.py](workload_basic.py) | 过滤、锁频、缓存控制、开销、导出、Source 页 |
| [02_memory_bound.py](02_memory_bound.py) | 合并 vs 非合并、float4、eager 3 个算子 vs 融合 |
| [03_register_bound.py](03_register_bound.py) | 寄存器 → Occupancy → 延迟隐藏；`__maxnreg__` spill；ILP vs TLP；软件预取 |
| [04_control_flow.py](04_control_flow.py) | if/else divergence；数据相关的循环次数与排序 |
| [05_bank_conflict.py](05_bank_conflict.py) | 矩阵转置：naive / shared memory / padding / swizzle |
| [06_tensor_core_roofline.py](06_tensor_core_roofline.py) + [06_plot_roofline.py](06_plot_roofline.py) | GEMM 的 dtype/对齐与 Tensor Core；本章所有 kernel 的 Roofline |
| [07_pipeline_kernels.py](07_pipeline_kernels.py) | 慢/快流水线 infer 阶段的全部 kernel |

> 为什么不用 `torch.utils.cpp_extension.load_inline`？它要编译 torch 头文件，并按 `TORCH_CUDA_ARCH_LIST` 生成 6 个架构，本机实测首次编译 **102 s**；本章 kernel 只用裸指针，直接 `nvcc -arch=sm_110 --shared` 只需约 5 s，编译选项也一目了然。在 ncu 里两者没有区别。

---

## 5.1 定位与工作原理

| | nsys（第 04 章） | ncu（本章） |
|---|---|---|
| 回答的问题 | GPU 在干什么、空闲在哪、kernel 之间的关系 | 单个 kernel 为什么慢 |
| 方式 | 低开销跟踪，所有 kernel 并发照常执行 | **串行化 + 重放**：每个 kernel 跑 N 遍（pass），每遍读一组计数器 |
| 开销 | 约 +30% | 每个 kernel 数百 ms 到数秒；本章 07 采 141 个 kernel 用了 **约 5 分钟** |
| 看不到 | kernel 内部 | kernel 之间的并发、重叠、空隙（练习 4） |

**replay 的含义**：一个 kernel 的指标分多次采集（`--set basic` 10 个 pass、`--set full` 40 个 pass，见 5.2），每个 pass 之前 ncu 会保存并恢复 kernel 写过的显存、默认还会**清空缓存**，保证每次重放的条件相同。所以 ncu 测到的 `Duration` 是"冷缓存下单独运行"的时间，不等于程序中真实的耗时。本章的 demo 都先在**不开 ncu** 的情况下用 CUDA Event 计时，再用 ncu 分析原因。

## 5.2 命令行采集

[01_ncu_basics.sh](01_ncu_basics.sh)，日志 [outputs/01_ncu_basics.log](outputs/01_ncu_basics.log)。

```bash
ncu --set full -o report python xxx.py                     # 最简单，但会采集所有 kernel（非常慢）
ncu -k regex:gemm --launch-skip 10 -c 1 ...                 # 名字匹配 gemm 的 kernel，跳过前 10 次，只采 1 次
ncu --nvtx --nvtx-include "measure/" ...                    # 只采 NVTX push/pop 范围 measure 内的 kernel（注意结尾的 /）
ncu --profile-from-start off ...                            # 只在 cudaProfilerStart/Stop 之间采集（= torch.cuda.profiler.start/stop）
ncu --kernel-name-base demangled -k regex:sum_functor ...   # 用完整签名（含模板参数）匹配
ncu --section SpeedOfLight --section Occupancy --metrics a,b,c ...   # 只采需要的 section / 指标
ncu -i report.ncu-rep --page raw --csv --metrics ...        # 导出指定指标
ncu -i report.ncu-rep --page source --print-source cuda,sass --csv   # Source 页（需要 -lineinfo + --import-source yes）
```

### 实测：过滤

| 命令 | 采到的 kernel |
|---|---|
| 不加过滤 | 12 个（包括 `torch.randn` 的 3 个 `distribution_elementwise_grid` 初始化 kernel 和 warmup 的 3 次 `x*2`） |
| `-k regex:elementwise --launch-skip 3 -c 2` | 第 4、5 次 `elementwise` |
| `--nvtx --nvtx-include "measure/"` | measure 范围内的 6 个（warmup 被排除） |
| `--kernel-name-base function -k regex:sum_functor` | `No kernels were profiled`（默认只匹配函数名 `reduce_kernel`） |
| `--kernel-name-base demangled -k regex:sum_functor` | 匹配到 `reduce_kernel<(int)512, ..., sum_functor<float, float, float>...>` |

本章的 demo 都用 NVTX 过滤：worker 先 warmup（在 NVTX 范围外），再在 `profile/<变体名>` 中把每个变体各跑一次；读报告时用 NVTX 栈的最内层作为变体名（`ncu_tools.load`）。

### 实测：开销

| 命令 | 时间 |
|---|---|
| 不开 ncu | 1.8 s |
| ncu 默认（basic set），12 个 kernel | 31.0 s |
| 只采 1 个 kernel：`--section SpeedOfLight` / `--set basic` / `--set detailed` / `--set full` | 10 / 10 / 21 / 40 个 pass，耗时 9.5 / 9.4 / 9.9 / 12.2 s |

ncu 有约 7 s 的固定启动开销（注入、初始化），之后每个 kernel 的开销与 pass 数成正比。小 kernel 的 pass 很快，大 kernel（例如 07 中的 GEMM）每个 pass 都要完整运行一次。

### 实测：缓存控制 `--cache-control`

8 MB 的 `x*2`（能放进 32 MB 的 L2），measure 范围内的第 3 次调用：

| | Duration | L2 命中率 | 从内存读入 L2 的 sector |
|---|---|---|---|
| `--cache-control all`（默认，每个 pass 前清空缓存） | 37.9 us | 0.08% | 262,434（= 8 MB） |
| `--cache-control none` | 38.3 us | 99.85% | 209 |

计数器的差异非常明确（L2 命中率 0.08% vs 99.85%），但这个 kernel 的 Duration 没有稳定的差别：脚本的第一次运行是 41.9 vs 37.4 us，最终日志是 37.9 vs 38.3 us，在运行间波动的范围内（8 MB 的 kernel 太小，启动和调度开销占了大头）。默认设置测的是"冷缓存"。如果一个小 kernel 在真实程序中总是紧跟在产生它输入数据的 kernel 后面（数据还在 L2 里），用 `none` 更接近真实情况；但这时结果会依赖前面跑过什么，可重复性变差。

### 实测：锁频 `--clock-control`

ncu 默认 `--clock-control base`，会把 GPU 锁在基础频率，让结果可复现。**在 Thor 上这个选项不生效**：两种设置下每个 kernel 都提示 `Data collection happened without fixed GPU frequencies`，SM Frequency 都是 1.57 GHz（宿主机的 `jetson_clocks` 已经把 GPU 锁在 1575 MHz），Duration 40.9 vs 40.3 us。结论：在 Jetson 上锁频靠 `nvpmodel` + `jetson_clocks`（第 01 章），而不是 ncu。

### 用 Python 读报告

Nsight Compute 自带 `ncu_report` 模块（`/opt/nvidia/nsight-compute/<版本>/extras/python`），本章所有对比表都用它生成：

```python
import ncu_report
ctx = ncu_report.load_report("outputs/02_memory_bound.ncu-rep")
a = ctx.range_by_idx(0).action_by_idx(0)                    # 第 1 个 kernel
a.name(), a.metric_by_name("gpu__time_duration.sum").as_double()
a.nvtx_state().domain_by_id(0).push_pop_ranges()            # ('profile', 'coalesced')
a.rule_results_as_dicts()                                   # 规则引擎的建议（type 2 = OPT）及预估加速
```

## 5.3 Thor 上的特殊之处

Thor 没有独立显存：GPU 通过 L2 → 内存控制器（MC）访问和 CPU 共享的 LPDDR5X，在 ncu 中这部分称为 **sysmem**。这带来三个问题：

**① 没有 `dram__*` 指标。** `--set full` 采到的 2306 个指标中，`dram__bytes_read.sum.pct_of_peak_sustained_elapsed` 等只能是 0 / n/a，Memory Workload Analysis Chart 会提示 `dram__bytes_read... could not be found`。

**② SOL 的 "Memory Throughput" 不是相对 DRAM 峰值的百分比。** 256 MB 的 `x*2`（读 268 MB + 写 268 MB，用时 2.37 ms ≈ 226 GB/s，已经接近第 00 章实测的 240 GB/s 峰值），SOL 却显示：

```
Memory Throughput    %   17.46      ← 实际上 DRAM 已经接近满载
L2 Cache Throughput  %   17.46
DRAM Throughput      （没有这一行）
```

并给出 "low compute throughput and memory bandwidth utilization ... typically indicate latency issues" 的**错误结论**。更糟的是，02 中慢 6.5 倍的 strided kernel 的 Memory% 是 35.5%，反而**高于**满带宽的 coalesced（17.0%）：Memory% 取的是 L1/L2 等各单元中最忙的那个，strided 的 L2 事务多，所以它高。

**③ 自己换算 DRAM 带宽**（[ncu_tools.py](ncu_tools.py)）：

| 方向 | 指标 | 说明 |
|---|---|---|
| 读 | `lts__d_sectors_fill_sysmem.sum` × 32 B | L2 未命中、从内存填充进 L2 的 sector，精确 |
| 写 | `min(lts__t_sectors_aperture_sysmem_op_write.sum, 4 × ..._op_write_lookup_miss.sum)` × 32 B | 估算，见下 |

写方向没有"写回 DRAM"的计数器。`..._op_write` 是 L2 收到的写请求 sector 数，非合并写会把同一个 sector 写多次：转置 naive 版的写请求是 2148 MB，是实际写入量 268 MB 的 8 倍，直接用它会得到"194% 峰值带宽"这种不可能的数字。实测发现 L2 写 miss 时按 128 B 整行（4 个 sector）分配：合并写的 miss 数 = 8.39 M（每个 sector 各 miss 一次），非合并写的 miss 数 = 2.10 M（= 8.39 M / 4）。所以 "4 × 写 miss 数" 是被写脏的不同 sector 数的上界，对本章所有 kernel 都给出 268.6 MB，等于逻辑写入量。利用率的分母用第 00 章实测的 240 GB/s（`ncu_tools.PEAK_BW_GBS`）。

> ⚠️ 这个写流量估计没有办法用 DRAM 计数器直接验证，对于"写入后很快被换出、又再次写入"的访存模式可能偏差较大。本章的表格同时列出读、写和原始写请求量，便于判断。

**④ 其他**：锁频不生效（5.2）；GPU 只有 20 个 SM，每个 SM 最多 48 个 warp（4 个 scheduler × 12），寄存器 64K 个，大 kernel 很容易出现 Tail Effect（5.9）。

## 5.4 Memory-bound：合并访存、向量化、融合

[02_memory_bound.py](02_memory_bound.py)，日志 [outputs/02_memory_bound.log](outputs/02_memory_bound.log)。N = 64M 个 float（256 MB），必需流量 = 读一次 x + 写一次 y = 537 MB。

| 变体 | 实现 | 计时 | 有效带宽 | DRAM%（换算） | Mem%（ncu） | sectors/请求（读） | L1→L2 流量 |
|---|---|---|---|---|---|---|---|
| torch_mul | `torch.mul(x, 2, out=y)` | 2.350 ms | 228.4 GB/s | 97 | 17.9 | 16.0 | 537 MB |
| coalesced | `y[i] = a*x[i]` | 2.441 ms | 219.9 GB/s | 92 | 17.0 | 4.0 | 537 MB |
| **strided** | 按列访问，相邻线程相差 8 MB | **15.904 ms** | **33.8 GB/s** | **14** | **35.5** | **32.0** | **2416 MB** |
| vec4 | float4 | 2.282 ms | 235.3 GB/s | 95 | 17.6 | 16.0 | 537 MB |
| eager_3ops | `relu(x*2+1)`：3 个 kernel | 7.103 ms | 75.6 GB/s | 92~95（每个） | ~17 | 16.0 | 3 × 537 MB |
| fused | 同样的计算，1 个 kernel | 2.444 ms | 219.7 GB/s | 91 | 17.0 | 4.0 | 537 MB |

（"计时"是不开 ncu、刷 L2 后的 CUDA Event 中位数；DRAM% 等来自 ncu。）

**读法**：

1. **合并访存**：一个 warp 的 32 个线程读连续的 32 个 float = 128 B = 4 个 sector，所以 `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum / l1tex__t_requests_...` = 4 是理想值（float4 是 16）。strided 是 32：每个线程单独占一个 sector，只用到其中 4/32 字节。注意它的 **DRAM 流量并没有增加**（读 268.8 MB，L2 命中率 86%，相邻列的数据还在 L2 里），慢是因为 L1→L2 的事务是 4.5 倍、L1 到 SM 的数据大部分被浪费。ncu 的提示是 Long Scoreboard 和 Drain stall。
2. **向量化**：vec4 和 torch_mul（PyTorch 的 `vectorized_elementwise_kernel` 也是每线程 4 个元素）的指令数只有 coalesced 的 1/4 ~ 1/6（13.6 M / 7.6 M vs 48.2 M），但 Memory-bound kernel 的瓶颈不在指令，所以只快了约 4~7%。向量化的收益在 SM% 上更明显（coalesced 16.3% → vec4 4.6%），这些 SM 资源可以留给融合进来的计算。
3. **融合是 ncu 发现不了的问题**：eager 的 3 个 kernel 单独看都达到了峰值的 92~95%，ncu 不会给出任何重要建议；但它们合起来读写了 1611 MB，是融合版的 3 倍，耗时 2.91 倍。**单 kernel 视角只能判断"这个 kernel 是否接近自己的上限"，判断不了"这个 kernel 该不该存在"**，这种问题要在 torch.profiler / nsys 层面发现（练习 3）。
4. 📎 笔记中 Memory-bound 的判据是 "Memory BW > 70% Peak"。在 Thor 上这里的 Peak 必须用换算后的 DRAM%，**不能用 SOL 的 Memory%**（满带宽的 kernel 只显示 17%）。笔记中"❌ 很难修复（内存带宽是硬件限制）"只对 DRAM 流量已经最小的 kernel 成立：合并访存（6.5x）、融合（2.9x）、低精度（练习 2）都是在减少流量或者让流量更有效。

## 5.5 Register-bound：Occupancy 低到底是不是问题

[03_register_bound.py](03_register_bound.py)，日志 [outputs/03_register_bound.log](outputs/03_register_bound.log)。

📎 笔记中的判据："Achieved Occupancy < 40% 且 Registers/Thread > 64 → Register-bound，修复：减少局部变量、`-maxrregcount`"。下面用两组实验检验它。

### 先看编译期信息

不用 ncu，`nvcc --resource-usage`（[kernels.py](kernels.py) 的 `resource_table()`）就能看到每个 kernel 的寄存器和 spill：

```
bmm8_per_thread            regs=128  spill_stores=0    B spill_loads=0
bmm8_per_thread_maxreg64   regs=64   spill_stores=1284 B spill_loads=1388 B   ← __maxnreg__(64)
bmm8_row_per_thread        regs=40
filter128_per_thread<0>    regs=168
filter128_per_thread<4>    regs=180                                          ← 预取多用了 12 个
filter128_split4<0/4>      regs=48
```

Occupancy 的计算（Thor：每 SM 65536 个寄存器、最多 48 个 warp）：128 regs × 32 线程 = 4096 个/warp → 最多 16 个 warp → 理论 Occupancy 16/48 = 33.3%。ncu 的 Occupancy 页中 `Block Limit Registers` 就是按这个算出来的"每个 SM 最多放几个 block"。

### A. 批量 8x8 矩阵乘法（每个线程内部 ILP 很高）

1M 个 8x8 矩阵，SoA 布局（`A[e][m]`，同一 warp 读同一个元素下标时地址连续），AI = 1.33 FLOP/B，是 Memory-bound 负载。机器人里批量的位姿变换、卡尔曼滤波都是这类计算。

| 变体 | regs | 理论/实测 Occ | 计时 | DRAM% | local memory 流量 | long scoreboard（周期/指令） |
|---|---|---|---|---|---|---|
| A_per_thread：一个线程算一个矩阵 | 128 | 33.3 / 31.2% | **4.312 ms** | **80** | 0 | 45.6 |
| A_per_thread_r64：同上 + `__maxnreg__(64)` | 64 | 66.7 / 62.6% | 12.982 ms（0.33x） | 69 | **2802 MB** | 187.9 |
| A_row_per_thread：一个线程算一行 | 40 | 100 / 98.1% | 4.815 ms（0.90x） | 69 | 0 | 51.6 |
| A_torch_bmm：`torch.bmm`（AoS） | 72 | 58.3 / 55.8% | 19.451 ms（0.22x） | 17 | 0 | 0.6 |

- Occupancy 只有 33% 的 A_per_thread **已经达到 80% 的 DRAM 带宽，是最快的**。每个线程在计算前一次性发出 128 个互相独立的 load，一个 warp 自己就能让大量访存同时在途（ILP），不需要很多 warp 来轮换。ncu 依然给出 "TheoreticalOccupancy：预估加速 66.7%" 的建议。
- 用 `__maxnreg__(64)`（对单个 kernel 的 `-maxrregcount`）把 Occupancy 提高一倍，结果**慢了 3 倍**（0.33x）：放不下的值被 spill 到 local memory（实际在 L1/L2/DRAM），产生了 2.8 GB 的额外流量，long scoreboard stall 增加到 4 倍（45.6 → 187.9 周期/指令）。
- 改变并行粒度后 Occupancy 达到 100%，反而慢了 10%：每个 B 元素被同一 block 的 8 个 warp 重复读取（L1 能命中，但 load 指令多了，LG Throttle stall 从 1.61 增加到 25.07 周期/指令）。
- `torch.bmm` 对这种"大量小矩阵"很不友好：被拆成 17 次 `magma_sgemmEx_kernel` 启动，SM% 79% 但大部分是索引和调度指令（19.5 亿条 warp 指令，是 A_per_thread 的 54 倍）。

### B. 滤波器组（每轮只有 1 个 load，ILP 低）

每个元素 128 个状态、沿时间维 256 步更新 `s[j] = s[j]*w_j + x_t`。每轮循环只有 1 个 load，后面 128 个 FMA 都依赖它，所以只能靠"别的 warp"或"提前发出的 load"来掩盖访存延迟。

| 变体 | regs | 理论 Occ | warps/scheduler | 就绪 warps/周期 | Issue% | FMA pipe% | 计时 |
|---|---|---|---|---|---|---|---|
| B_per_thread | 168 | 25.0% | 2.96 | 0.66 | 37 | 35 | 6.321 ms |
| B_split4：4 个线程分担 128 个状态 | 48 | 83.3% | 9.85 | 1.05 | 42 | 34 | 6.357 ms（**0.99x**） |
| B_prefetch：提前 4 轮发出 load | 180 | **16.7%** | 1.97 | 0.75 | 62 | 57 | 3.912 ms（**1.62x**） |
| B_split4_prefetch | 48 | 83.3% | 9.81 | 4.29 | 88 | 63 | 3.808 ms（1.66x） |

- B_per_thread 是真的被延迟卡住了：long scoreboard 6.09 周期/指令，Issue 只有 37%。ncu 给出 "TheoreticalOccupancy（25%，受寄存器限制）：预估加速 63.1%"。
- **但按建议把 Occupancy 提到 83%（B_split4）完全没有加速。** 原因可以用 Little 定律解释：要掩盖延迟，需要"在途的工作量" ≥ 延迟 × 吞吐。B_per_thread 每个 scheduler 有 2.96 个 warp × 每轮 128 个 FMA ≈ 379；B_split4 有 9.85 个 warp × 每轮 32 个 FMA ≈ 315。warp 多了 3.3 倍，每个 warp 两次 load 之间的工作却只剩 1/4，总量没有变化。表现在 stall 上：每条指令的 long scoreboard 等待从 6.09 增加到 20.55 周期。
- **软件预取**（只多用 12 个寄存器，Occupancy 反而降到 16.7%）让下一轮的 load 和这一轮的计算重叠，Issue 从 37% 提高到 62%，快了 1.62 倍。这是增加 ILP，而不是增加 TLP。
- Source 页（`T.hot_lines`）印证了这一点：B_per_thread 91.0% 的 stall 采样落在 FMA 那一行（第 139 行，等 `v`），load 所在行只占 6.8%；预取之后 FMA 行的占比降到 31.8%。**stall 会记在等待数据的那条指令上，而不是发起 load 的那条**，读 Source 页时要往前找数据的来源。

### 结论

1. Occupancy 是**手段**，不是目标。需要的是"足够多的在途工作"，TLP（更多 warp）和 ILP（每个 warp 更多独立指令）都能提供。判断是否真的受限，要看 **Issue Slot 利用率、Eligible Warps、stall 原因**，以及改动之后的实测耗时。
2. `-maxrregcount` / `__maxnreg__` / `__launch_bounds__` 只有在"少几个寄存器就能多放一个 block、并且不 spill"时才有用；先看 `--resource-usage` 的 spill 字节数，不为 0 的话通常是负收益（A 组实测慢 3 倍）。
3. ncu 规则的"预估加速"是在其他条件不变的假设下算出的上限，而改变 Occupancy 的手段几乎总会改变其他条件（练习 6）。

## 5.6 Control-flow-bound：warp divergence

[04_control_flow.py](04_control_flow.py)，日志 [outputs/04_control_flow.log](outputs/04_control_flow.log)。

📎 笔记：warp 内的线程走不同分支时，两条路径**串行执行**，耗时 = A + B；`for` 循环次数不一致也一样。判据："IPC < 1.5、Branch Efficiency < 90%"。

| 变体 | 计时 | IPC（满分 4） | Issue% | 活跃线程/warp | Branch Efficiency | warp FMA 指令 | 线程 FMA | FMA 有效率 |
|---|---|---|---|---|---|---|---|---|
| A_divergent：`if (threadIdx.x & 1)` | 1.627 ms | **3.99** | 100 | **16.2** | **99.4%** | 137 M | 2.15 G | **49%** |
| A_warp_uniform：`if ((threadIdx.x >> 5) & 1)` | 0.831 ms（1.96x） | 3.97 | 99 | 32.0 | 100% | 68 M | 2.15 G | 98% |
| B_random：每个元素迭代 cnt ~ U[0,1024) 次 | 1.359 ms | 3.94 | 99 | 16.7 | 59.5% | 132 M | 2.15 G | 51% |
| B_sorted：先按 cnt 排序 | 0.716 ms（1.90x） | 3.83 | 96 | 32.0 | 100% | 68 M | 2.15 G | 99% |
| B_uniform_mean：都迭代平均次数（下限） | 0.706 ms | 3.91 | 98 | 32.0 | 100% | 68 M | 2.15 G | 99% |

（"FMA 有效率" = 线程级 FFMA 数 / (32 × warp 级 FMA 指令数)。）

1. **"IPC < 1.5" 这个判据在这里完全失效**：divergent kernel 的 IPC 是 3.99、Issue 100%、SM 99%，硬件看起来满负荷，但每条指令只有 16.2 个线程在干活，一半的算力被浪费。divergence 浪费的是 **warp 内的 lane**，而不是 issue slot；IPC 统计的是 warp 指令，所以看不出来。（笔记中"IPC 低"的四种原因里，只有内存延迟和同步会真正降低 IPC。）
2. **Branch Efficiency 也会误导**：A_divergent 有 99.4%。它统计的是"分支指令中所有活跃线程方向一致的比例"，而这个 kernel 只有一次分歧的 if，其余都是方向一致的循环分支。它反映不了分歧之后两条路径各执行了多久。
3. **可靠的指标**：Warp State Statistics 中的 `Avg. Active Threads Per Warp`（`smsp__thread_inst_executed_per_inst_executed.ratio`，满分 32），以及线程级 / warp 级指令数之比。ncu 的 `ThreadDivergence` 规则就是用它给出 "预估加速 49%" 的，和实测的 1.96x 吻合。
4. 数据相关的循环：随机顺序时每个 warp 的 cnt 最大值平均是 992，而均值只有 512，所以 warp 平均要跑 992 轮；排序后两者相等。**但 `torch.sort` 4M 个元素要 2.25 ms，比节省的 0.64 ms 还多**：只有当同一份排序结果能被多次复用（例如 NMS 前按分数排序、光线追踪按材质分组），或者排序可以和其他计算融合时，这种修复才划算。
5. 📎 笔记中的修复手段："重构 kernel 避免分支""用 warp 级原语""提前退出"。本例的修复本质是**让分支条件以 warp 为单位一致**（A：改变线程到数据的映射；B：重排数据）。编译器能把短分支转换成无分支的 select（predication），所以写 demo 时两个分支故意用了不同的循环结构，避免被优化掉（[cuda/bottlenecks.cu](cuda/bottlenecks.cu) 中 `path_a/path_b` 的注释）。

## 5.7 Shared memory bank conflict

[05_bank_conflict.py](05_bank_conflict.py)，日志 [outputs/05_bank_conflict.log](outputs/05_bank_conflict.log)。8192×8192 float32 矩阵转置，必需流量 537 MB。

📎 笔记：shared memory 分成 32 个 bank（每个 4 字节宽，`bank = (地址/4) % 32`），同一 warp 的多个线程访问**同一 bank 的不同地址**时访问被串行化；访问同一地址则是广播，不冲突。

```cuda
__shared__ float tile[32][32];
tile[threadIdx.y + j][threadIdx.x] = in[...];        // 写：同一 warp 的 threadIdx.x 连续 → 32 个不同 bank ✓
out[...] = tile[threadIdx.x][threadIdx.y + j];       // 读：相邻线程相差一行 = 32 个 float → 全部落在同一个 bank ✗ 32-way
__shared__ float tile[32][33];                       // padding：(x*33 + y) % 32 = (x + y) % 32，错开 ✓
tile[row][col ^ row]                                 // swizzle：不浪费空间，同样错开 ✓
```

| 变体 | 计时 | 有效带宽 | 全局写 sectors/请求 | shared load 的 wavefront/指令 | bank conflict（load） | MIO Throttle（周期/指令） |
|---|---|---|---|---|---|---|
| copy（参照：不转置，上限） | 2.479 ms | 216.6 GB/s | 4.0 | - | - | 0 |
| naive：直接转置 | 5.432 ms | 98.8 GB/s | **32.0** | - | - | 58.6 |
| smem：tile[32][32] | 3.680 ms（1.48x） | 145.9 GB/s | 4.0 | **32.4** | **65.8 M** | 34.1 |
| smem_pad：tile[32][33] | 3.418 ms（1.59x） | 157.1 GB/s | 4.0 | 1.0 | 0.06 M | 0 |
| smem_swizzle | 3.431 ms（1.58x） | 156.5 GB/s | 4.0 | 1.0 | 0.06 M | 0 |
| `inp.t().contiguous()`（PyTorch） | 9.444 ms | 56.8 GB/s | | | | |

1. **wavefront** 是 shared memory 处理一次请求所需的"拍数"：无冲突时一条 warp 级 load 只需 1 个 wavefront，32-way 冲突需要 32 个。ncu 直接报告了 "32.4-way bank conflict"，规则 `UncoalescedSharedAccess` 给出 "65,011,712 excessive wavefronts（94%）"。
2. **Source 页精确定位**到第 273 行（按列读 tile 的那一行）：`L1 Wavefronts Shared` = 67.1 M，而 `Ideal` = 2.1 M。修复后的版本该指标为 0。命令行写法见 `ncu_tools.hot_lines()`：`ncu -i rep --page source --print-source cuda,sass --csv`。
3. **消除 bank conflict 只快了 7%**（3.680 → 3.418 ms）：shared 版本真正的大收益来自把**全局写变成合并写**（naive → smem 1.48x），bank conflict 是第二位的问题。修复后仍然只有 copy 的 73%，剩下的差距主要是 `__syncthreads()`（barrier stall 约 20 周期/指令）和每个 block 只处理一个 tile 导致的启动/尾部开销。**修复前先看它占了多少**：bank conflict 虽然在 ncu 中非常醒目（规则给出预估加速 91%），实际只占 7%。
4. 注意 naive 的写：L2 收到了 2148 MB 的写请求（8 倍于实际写入量），这正是 5.3 节需要修正写流量估计的原因。
5. PyTorch 的 `.t().contiguous()` 在这里比 naive 还慢（9.44 ms）：它走的是通用的 strided copy kernel。大矩阵转置是热点时，值得手写或者用 Triton 实现。

## 5.8 Tensor Core 与 Roofline

[06_tensor_core_roofline.py](06_tensor_core_roofline.py)，日志 [outputs/06_tensor_core.log](outputs/06_tensor_core.log)。GEMM M=N=K=4096。

| 变体 | cuBLAS 选的 kernel | 计时 | TFLOP/s | tensor pipe% | tc pipe% | FMA pipe% | Tensor Core 运算 / 2MNK |
|---|---|---|---|---|---|---|---|
| fp32（关闭 TF32） | `Kernel2` | 22.209 ms | 6.2 | 0 | 0 | 79.1 | 0（FP32 运算 = 1.000） |
| tf32 | `cutlass3x_sm100_tensorop_s256x256x...` | 2.623 ms | 52.4 | 41.3 | 44.2 | 0.2 | 1.000（`tf32_dst_fp32`） |
| fp16 | `nvjet_sm110_hsh_256x256_64x4_2x1_2cta...` | 1.065 ms | 129.0 | 55.7 | 57.3 | 0.4 | 1.000（`fp16_dst_fp32`） |
| bf16 | `nvjet_sm110_tst_256x256_64x4_2x1_2cta...` | 0.949 ms | 144.9 | 58.2 | 59.8 | 0.5 | 1.000（`bf16_dst_fp32`） |
| **fp16，K=4095** | **`Kernel2`** | **6.914 ms** | **19.9** | 62.6 | **0** | 0.6 | 1.008 |

- **怎么确认 GEMM 用了 Tensor Core**：`sm__ops_path_tensor_op_{hmma,utchmma}_src_<输入>_dst_<累加>.sum`，它按 FLOP 计数（实测正好等于 2·M·N·K = 137,438,953,472），能同时告诉你输入和累加的精度。fp32 GEMM 的这些指标全部为 0，FP32 运算数正好是 2MNK。
- **对齐的影响**：K=4095（不是 8 的倍数）时，cuBLAS 从 `nvjet`（Blackwell 的 tcgen05 指令，`tc pipe` 有活动）退回到通用的 `Kernel2`，后者仍然用 Tensor Core（`tensor pipe` 62.6%，`hmma` 运算 1.008 倍于 2MNK，多出来的是补齐的部分），但走的是旧的 mma 路径，`tc pipe` 为 0，**慢了 6.5 倍**。所以"用了 Tensor Core"还不够，还要看走的是哪条路径；最简单的办法是把维度 padding 到 8（最好 64）的倍数。
- 这些 GEMM kernel 的 Occupancy 都很低（每线程 206~255 个寄存器、200 KB 以上的 shared memory，每 SM 只放 1 个 block），但都接近各自的屋顶，又一次说明 Occupancy 不是目标（5.5）。
- fp32 → fp16 的 22.2 → 1.07 ms（约 21 倍）就是第 07 章"低精度"对 Compute-bound 算子的收益来源。

### Roofline：本章所有 kernel

[06_plot_roofline.py](06_plot_roofline.py) 读取 02~06 的报告：横轴 AI = ncu 统计的 FLOP（`2×FFMA + FADD + FMUL`，或 Tensor Core 运算数）/ 5.3 节估算的 DRAM 字节，纵轴 = FLOP / ncu 测得的 kernel 时间；屋顶是第 00 章实测的 240 GB/s、6.39 TFLOPS（FP32）、155.6 TFLOPS（FP16 Tensor Core）。05 的转置没有浮点运算，不在图中。

![roofline](outputs/roofline_ncu.png)

- 02 的逐元素 kernel 都**贴在带宽斜线上**，只有 strided 掉到线下 7 倍：Memory-bound 的 kernel 离斜线有多远，就说明访存效率有多差。融合让点**向右移**（AI 从 0.08 到 0.25），这是第 00 章讲过的"算子融合减少访存"。
- `A_per_thread_r64` 因为 spill 产生了额外的 DRAM 流量，点**向左下方**移动（AI 从 1.33 降到 0.50）。
- 03-B、04 的 kernel 位于拐点（约 27 FLOP/B）右侧，属于 Compute-bound，但离 FP32 屋顶还有 1.5~2.4 倍：延迟（03-B）和 divergence（04）都表现为"在屋顶下方"，修复后向上移动。**Roofline 只告诉你离上限有多远，不告诉你原因**，原因要看 5.5/5.6 中的 stall 和活跃线程数。
- GEMM：fp16/bf16 贴近 Tensor Core 屋顶，K=4095 的 fp16 掉下来 7 倍。

## 5.9 分析流水线中的真实 kernel

[07_pipeline_kernels.py](07_pipeline_kernels.py)，日志 [outputs/07_pipeline_kernels.log](outputs/07_pipeline_kernels.log)。采集方式：`--profile-from-start off`（worker 在 warmup 5 步后调用 `torch.cuda.profiler.start()`，只跑 1 步）+ `--nvtx-include "infer/"`。每个配置约 140 个 kernel，**ncu 用了约 5 分钟**（不开 ncu 时一步只要约 10 ms）。

**慢版（fp32）**：infer 阶段 141 个 kernel、23 种，GPU 时间合计 4573 us（ncu 冷缓存下测得）。

| kernel | 次数 | 占比 | grid × block | SM% | DRAM% | Occ% | regs | 说明 |
|---|---|---|---|---|---|---|---|---|
| `Kernel2`（cuBLAS fp32 GEMM） | 19 | **47.7%** | 64 × 128 | 29.9 | 23 | 15.6 | 198 | ncu：理论 Occupancy 16.7%（受寄存器限制），**Tail Effect** |
| `nchwToNhwcKernel` | 17 | 8.5% | 8192 × 256 | 55.6 | 49 | 90.2 | 24 | 布局转换，第 04 章 nsys 中占 6.9% |
| `vectorized_elementwise_kernel` | 29 | 7.8% | | 8.5 | 66 | 97.3 | 16 | 逐元素 |
| `sgemm_largek_lds64` | 6 | 5.9% | | 50.8 | 14 | 92.2 | 39 | |
| `sm80_xmma_fprop_implicit_gemm_tf32...`（cuDNN 卷积） | 3+5 | 7.6% | | 17~22 | | 8.5 | 96~128 | tensor pipe 21~25% |
| `fmha_cutlassF_f32_aligned_64x64_rf_sm80` | 6 | 3.4% | 16 × 128 | 12.7 | 22 | 8.4 | 168 | fp32 attention |

- 最大的问题是 fp32 GEMM：每个 SM 只能放 2 个 block（198 regs × 128 线程），一轮 40 个 block，而 grid 只有 64 个 block，**第二轮只有 24 个 block，40% 的 block 槽位空着**，这就是 ncu 提示的 Tail Effect（预估加速 50%）。（推测，未验证：ToyVLA 在 batch=1 时矩阵很小，fp32 下 cuBLAS 的 kernel 选择有限。）
- 修复方向（第 07/08 章验证）：fp16（换成 nvjet/cutlass 的 Tensor Core kernel）、channels_last（消除 nchwToNhwc），这正是快版做的事。

**快版（fp16）**：136 个 kernel，GPU 时间合计 1906 us（慢版的 42%）。

| kernel | 次数 | 占比 | SM% | DRAM% | tensor% | tc% | Occ% |
|---|---|---|---|---|---|---|---|
| `vectorized_elementwise_kernel` | 40 | **19.4%** | 4.1 | 14 | 0 | 0 | 62.4 |
| `batch_norm_transform_input_channels_last` | 12 | 10.4% | 17.7 | 15 | 0 | 0 | 82.9 |
| `elementwise_kernel` | 9 | 9.6% | 40.8 | 44 | 0 | 0 | 88.8 |
| `Kernel2` | 7 | 7.0% | 18.6 | 50 | 29 | 0 | 14.0 |
| `vectorized_layer_norm_kernel` | 13 | 7.0% | 11.7 | 10 | 0 | 0 | 37.9 |
| `nvjet_sm110_hsh_64x56_...`（GEMM） | 6 | 5.9% | 11.3 | 59 | 19 | 25 | 13.1 |
| `flash_fwd_kernel` | 6 | 5.1% | 3.8 | 13 | 12 | 0 | 8.3 |

- fp16 之后，GEMM 已经不是主要耗时项了；**逐元素、BN、LayerNorm 这些小 kernel 加起来超过 45%**，其中 vectorized_elementwise、BN、LayerNorm 的 SM% 只有 4~18%、DRAM% 只有 10~15%：每个 kernel 的数据量太小（batch=1），连带宽都用不满。这类问题的修复不在单个 kernel 内部，而是**融合**（torch.compile，第 03 章实测 reduce-overhead 2.96x）或者 TensorRT（第 06 章）。
- 这是"自顶向下"的又一个例子：nsys 说快版仍然是 Launch-bound（第 04 章：GPU busy 21%），ncu 进一步说明了**就算没有 launch 开销，这些 kernel 自身的效率也很低**。

## 5.10 按 Section 解读报告：速查

| Section | 看什么 | 对应的瓶颈 / 本章例子 | Thor 上的注意事项 |
|---|---|---|---|
| **GPU Speed Of Light** | Compute (SM) % vs Memory % | 初步判断 Compute / Memory / 延迟 | ⚠️ Memory% 不是 DRAM%（5.3），要自己换算 |
| **Roofline Chart** | 离屋顶多远 | 5.8 | 图中的带宽屋顶基于 DRAM 指标，Thor 上不可靠；用 5.8 的方法自己画 |
| **Launch Statistics** | grid/block、regs、shared memory、Waves Per SM | Tail Effect（5.9） | 20 个 SM，小 grid 很容易出现尾部效应 |
| **Occupancy** | 理论 / 实测 Occupancy、Block Limit Registers/Shared Mem/Warps | Register-bound（5.5） | 每 SM 最多 48 个 warp |
| **Scheduler Statistics** | Issue Slot、Eligible Warps | 延迟是否被隐藏（5.5） | |
| **Warp State Statistics** | stall 原因、Avg. Active Threads Per Warp | Long Scoreboard（访存延迟）、MIO Throttle（shared memory）、Barrier、divergence（5.6） | |
| **Memory Workload Analysis** | L1/L2 命中率、sectors/请求、bank conflict、spill | 合并访存（5.4）、bank conflict（5.7）、local memory（5.5） | 没有 DRAM 一栏，L2 ↔ System Memory 的流量要按 5.3 换算 |
| **Compute Workload Analysis** | IPC、各流水线利用率 | Tensor Core（5.8） | |
| **Instruction Statistics** | 指令数、指令构成 | 向量化（5.4） | |
| **Source Counters** + Source 页 | 每行源码 / SASS 的 stall 采样、wavefront | 定位到行（5.5、5.7） | 需要 `-lineinfo` + `--import-source yes` |

**stall 原因速查**（📎 笔记中的 "Register Dependency""Branch Divergence""LG Throttle（寄存器和共享内存竞争）"在 ncu 中并不存在或含义不同，以下是 ncu 实际的名字）：

| stall | 含义 | 本章出现的地方 |
|---|---|---|
| Long Scoreboard | 等 L1TEX（global/local/texture）的数据 | 所有 Memory-bound kernel、03 的 spill |
| Short Scoreboard | 等 MIO（shared memory、特殊函数）的数据 | 05 smem |
| MIO Throttle | MIO 指令队列满（shared memory 请求太多，比如 bank conflict） | 05 smem（34 周期/指令） |
| LG Throttle | Local/Global 访存指令队列满 | 03 row_per_thread、spill |
| Barrier | 等 `__syncthreads()` | 05 shared memory 版本 |
| Drain | kernel 结束时等待写回完成 | 02 strided |
| Wait | 固定延迟的依赖 | |
| Math Pipe Throttle | 计算流水线忙 | |

## 常见坑

1. **在 Thor 上把 SOL 的 Memory% 当作 DRAM 利用率**：满带宽的 kernel 只显示约 17%，ncu 还会据此给出"延迟问题"的错误结论（5.3）。
2. **直接用 L2 写请求量算 DRAM 写流量**：非合并写会被重复计数，转置 naive 版算出 194% 的峰值带宽（5.3）。
3. **不加过滤就采集**：流水线一步 140 个 kernel 要 5 分钟；模型稍大就是几小时。总是用 `-k`、`-c`、`--nvtx-include`、`--profile-from-start off`。
4. **用 ncu 的 Duration 当真实耗时**：ncu 默认清空缓存、串行化、锁频（Thor 上锁频不生效），测的是"冷启动下单独运行"的时间。耗时要在不开 ncu 时用 CUDA Event 测（本章每个 demo 的第 1 步）。
5. **照着规则引擎的"预估加速"去优化**：03-B 中提高 Occupancy 的预估加速是 63%，实测 0.99x；05 中 bank conflict 的预估加速 91%，实测 7%。
6. **用 `-maxrregcount` 提高 Occupancy**：先看 `--resource-usage` 的 spill，有 spill 通常是负收益（03-A 慢 3 倍）。
7. **用 IPC 或 Branch Efficiency 判断 divergence**：divergent kernel 的 IPC 是 3.99、Branch Efficiency 99.4%，都看不出问题；要看 Avg. Active Threads Per Warp（5.6）。
8. **在 Source 页把 stall 归咎于它所在的那一行**：stall 记在等待数据的指令上，要往前找数据来自哪条 load（5.5）。
9. **以为 ncu 能分析并发**：kernel 被串行化重放，看不到 overlap、stream 并发和 kernel 之间的空隙，这些用 nsys（练习 4）。
10. **demo 被编译器"优化掉"**：两个分支结构相同时可能被转换成无分支代码；寄存器用量依赖常量能否折叠（03 的 `filter_bank` 把系数下标做成模板参数，寄存器从 72 降到 48）。写 demo 之后先看 `--resource-usage` 和 SASS。

## 练习

见 [exercises.md](exercises.md)。

## 小结

- ncu 回答"这个 kernel 为什么慢"。流程：**先用 nsys/torch.profiler 找到值得看的 kernel → 不开 ncu 计时 → 用过滤选项采集 → SOL 判断大类 → 对应的 Section 找原因 → Source 页定位到行 → 修改后对比**。
- Thor 特有：没有 DRAM 指标，SOL 的 Memory% 有误导，需要用 L2 sysmem 计数自己换算；`--clock-control` 不生效，锁频靠 jetson_clocks。
- 四类瓶颈的实测（修复前 → 修复后）：

  | 瓶颈 | 关键指标 | 修复 | 收益 |
  |---|---|---|---|
  | 非合并访存 | sectors/请求 32（理想值 4） | 合并访存 | 6.5x |
  | 多余的 kernel | ncu 看不出（每个 kernel 都满带宽） | 融合 | 2.9x |
  | 访存延迟 + 低 ILP | Issue 37%、Long Scoreboard | 软件预取（Occupancy 反而更低） | 1.62x |
  | 提高 Occupancy | 理论 Occupancy 25% → 83% | 4 个线程分担 | **0.99x** |
  | 限制寄存器 | spill 2.8 GB | `__maxnreg__(64)` | **0.33x** |
  | warp divergence | 活跃线程 16.2/32（IPC 仍是 3.99） | 以 warp 为单位分支 / 排序 | 1.96x / 1.90x |
  | bank conflict | 32.4 wavefront/指令 | padding / swizzle | 1.07x（合并写贡献了 1.48x） |
  | 维度不对齐 | `tc pipe` 0 | padding 到 8 的倍数 | 6.5x |

- 笔记中的经验阈值（"Occupancy < 40%""IPC < 1.5""Branch Efficiency < 90%"）在本章的实测中多次失效，它们是**线索**，判断依据是改动之后的实测耗时。
- 下一步：第 06 章 TensorRT 的三步法中，ncu 是第三步，用来分析 TensorRT 选出的 kernel。
