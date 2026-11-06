# 第 00 章 基础理论：性能瓶颈与度量

> 📎 对应笔记：[doc/CUDA.xmind](../doc/CUDA.xmind) →「硬件架构」「性能分析·第一部分（利用率）」「第二部分（四大瓶颈）」「Roofline model」「STREAM 和 EVENT」
>
> 本章所有数字都在 **Jetson AGX Thor（MAXN，GPU 锁频 1575 MHz）** 上实测得到，日志在 [outputs/](outputs/)。

## 学习目标

学完本章你应该能：

1. 说出 6 类常见瓶颈（CPU、同步、Launch、Memory、Register、Control-flow），以及判断每一类分别要看什么指标、用什么工具；
2. 分清"GPU 利用率"的几层含义，知道为什么"利用率 97%"可能只用了 11% 的算力；
3. 在自己的机器上实测 Roofline，判断一个算子是 memory-bound 还是 compute-bound；
4. 写出**正确**的 GPU 计时代码，避开冷启动、异步、L2 缓存、尾延迟这几个坑。

## 目录

- [0.1 为什么要先 profile](#01-为什么要先-profile)
- [0.2 GPU 执行模型速览](#02-gpu-执行模型速览)
- [0.3 GPU"利用率"指标体系](#03-gpu利用率指标体系)
- [0.4 Roofline 模型（实测）](#04-roofline-模型实测)
- [0.5 瓶颈类型与诊断方法](#05-瓶颈类型与诊断方法)
- [0.6 正确的计时方法](#06-正确的计时方法)
- [0.7 共享工具 common/](#07-共享工具-common)
- [常见坑](#常见坑) · [练习](exercises.md) · [小结](#小结)

| Demo | 内容 | 运行时间 |
|---|---|---|
| [01_timing_pitfalls.py](01_timing_pitfalls.py) | 计时的 5 个坑 | ~10 s |
| [02_roofline.py](02_roofline.py) | 实测峰值带宽和算力，画 Roofline | ~60 s |
| [03_mfu.py](03_mfu.py) | FLOPs 统计、MFU 和 GPU busy 的对比 | ~60 s |

```bash
cd /root/data/learn/profile
/opt/venv/bin/python 00_theory/01_timing_pitfalls.py
/opt/venv/bin/python 00_theory/02_roofline.py
/opt/venv/bin/python 00_theory/03_mfu.py
```

---

## 0.1 为什么要先 profile

**Amdahl 定律**：如果某部分占总耗时的比例是 `p`，把它加速 `s` 倍，整体加速比为

```
Speedup = 1 / ((1 - p) + p / s)
```

即使把一个只占 10% 的 kernel 优化到无限快，整体也最多快 1.11 倍。所以优化的第一步永远是**测量**：先找到占比最大的那一块，再判断它属于哪类瓶颈，最后选对应的手段。

我们的慢流水线（[common/pipeline.py](../common/pipeline.py)）就是一个现成的例子。下面是 `python common/selftest.py` 的实测结果：逐个关闭低效开关，看每个开关对延迟的影响：

```
name                            p50(ms)    p90(ms)    p99(ms)  speedup
----------------------------------------------------------------------
slow (all on)                    11.371     11.681     12.008    1.00x
slow - py_preprocess              8.531      8.941      9.204    1.33x
slow - sync_in_loop               9.618      9.718     10.215    1.18x
slow - no_pin                    10.994     11.454     11.477    1.03x   <- 凭直觉以为很慢，实测影响很小
slow - fp32                       9.376      9.467      9.797    1.21x
slow - eager_small_ops            7.529      7.952      7.983    1.51x   <- 最大的单项瓶颈
slow - realloc                   10.252     10.356     10.415    1.11x
fast (all off)                    3.341      3.526      3.630    3.40x
```

`no_pin`（不用 pinned memory）在教科书里通常被列为常见瓶颈，但在 Thor 上几乎不影响延迟：Thor 的 CPU 和 GPU 共享同一块物理 DRAM，而且这里要拷贝的数据只有 600 KB。**凭直觉判断的瓶颈经常是错的**，这就是要先测量再优化的原因。

## 0.2 GPU 执行模型速览

CUDA 硬件基础（SM、Warp、内存层级、bank conflict 等）在 [CUDA 笔记](../doc/CUDA.xmind) 里已有详细整理，这里只列出和性能分析直接相关的结论。

**CPU 和 GPU 是异步协作的**：CPU 调用 `a @ b` 只是把 kernel 放进 stream 队列就立即返回，GPU 稍后才真正执行。这一点决定了：
- Python 层的计时、VizTracer 等工具看到的函数耗时，**不等于** GPU 执行时间（见 0.6）；
- 当 CPU 下发 kernel 的速度跟不上 GPU 执行速度时，GPU 就会空等，这就是 Launch-bound（见 0.5）。

**本机规格**（`torch.cuda.get_device_properties` + jtop，详见 [plan/env_report.md](../plan/env_report.md)）：

| 项 | 值 | 对性能分析的意义 |
|---|---|---|
| SM 数 | 20 | 一个 kernel 至少要有 20 个 block 才能占满所有 SM |
| 每 SM 最大线程 / 寄存器 | 1536 / 65536 | 每线程用 64 个寄存器时，每 SM 最多驻留 1024 个线程（Occupancy 67%） |
| 每 SM shared memory | 228 KB | |
| **L2 cache** | **32 MB** | 很大！32 MB 以内的数据反复访问会命中 L2，测出来的带宽会虚高（见 0.4） |
| 内存 | 128 GB LPDDR5X，CPU/GPU **统一内存** | H2D 拷贝不经过 PCIe；CPU 和 GPU 争用同一份带宽 |
| GPU 频率 | 1575 MHz（MAXN + jetson_clocks 锁定） | |

**Jetson 特有的注意事项**：
- 统一内存：`pin_memory` 和 `non_blocking` 的收益比独显平台小得多（0.1 节的实测已经证明这一点）；CPU 密集的预处理会和 GPU 抢占 EMC 带宽。
- DVFS 与功耗墙：benchmark 之前必须确认功耗模式（`nvpmodel`）和锁频状态（`jetson_clocks`）。即使频率已锁定，持续满载时也可能撞上功耗墙（见 0.4 的实测）。

## 0.3 GPU"利用率"指标体系

"GPU 利用率"至少有 7 种不同的含义，它们回答的问题各不相同：

| 层次 | 指标 | 回答的问题 | 本机获取方式 |
|---|---|---|---|
| 功耗 | GPU rail 功耗 | 是否真的在"出力" | jtop（`VDD_GPU`） |
| 时间维度 | GPU Util（`nvidia-smi dmon` 的 `sm` 列，NVML `utilization.gpu`） | 采样周期内有多少时间**至少有一个 kernel** 在跑 | nvidia-smi / NVML / jtop |
| 空间维度 | SM Active / SM Efficiency | 忙的时候，有多少比例的 SM 在干活 | nsys `--gpu-metrics-devices`（需要计数器权限，privileged 容器中可用；实测见[第 04 章 4.8](../04_nsight_systems/README.md#48-gpu-metrics空间维度的利用率拷贝由谁执行)） |
| Warp 维度 | Achieved Occupancy | 一个 SM 内实际驻留了多少 warp | ncu（需要权限；见[第 05 章](../05_nsight_compute/README.md)） |
| Tensor Core | Tensor Active | 矩阵乘法有没有走 Tensor Core | nsys GPU metrics / ncu |
| 带宽 | `dmon` 的 `mem` 列 | 显存控制器**忙碌的时间占比**（不是容量占用，也不是"实际 GB/s ÷ 峰值"） | nvidia-smi（⚠️ Thor 上始终为 0）；nsys GPU metrics 和 ncu 在 Thor 上也没有 DRAM 指标，ncu 需按第 05 章 5.3 自己换算 |
| 有效产出 | **MFU / HFU** | 实际有效 FLOPs ÷ 理论峰值 | 根据模型 FLOPs 手算（见下面的 demo） |

**核心认知**：上表除了 MFU/HFU，其余指标都只说明硬件"忙不忙"，不说明"产出了多少"。利用率高不代表算力用满；利用率低则一定没用满。

> 实测（第 04 章 4.8，慢流水线的 infer 阶段）：GR Active（时间维度）88% → SMs Active（空间维度）76% → SM Issue（真正发射指令）29% → Tensor Active 2%，逐层下降。

### MFU 的计算

```
MFU = (每次迭代的模型 FLOPs / 每次迭代耗时) / GPU 峰值 FLOPs
```

- 分子的 FLOPs 是"数学上必须做的计算量"，由模型结构决定，和实现方式无关。经验公式：
  - Transformer **推理**前向 ≈ `2 × N × tokens`（N 为参数量）；**训练** ≈ `6 × N × tokens`（前向 1 份 + 反向 2 份）
  - attention 的 `QK^T` 和 `AV` 不在参数里，序列较长时要额外加上 `4 × L × T² × d`（L 为层数）
  - CNN 需要逐层计算，或者直接用工具统计
- HFU 的分子是"硬件实际执行的 FLOPs"（包括 activation checkpointing 的重计算），所以 HFU ≥ MFU。
- 工具：`torch.utils.flop_counter.FlopCounterMode`。

### 实测：GPU busy vs MFU（[03_mfu.py](03_mfu.py)）

分母使用 0.4 节实测的峰值（FP16 156.8 TFLOPS，FP32 6.39 TFLOPS）；GPU busy 通过 NVML 采样得到（等价于 `nvidia-smi` 的 sm%）：

```
[公式验证] TinyTransformer T=256: 2×N×T = 6.44 GFLOPs，+ attention 4×(2·T²·d)×2 = 0.54，合计 6.98；FlopCounter = 6.98 GFLOPs
[坑] 不关 fastpath 时 FlopCounter = 0.00 GFLOPs（漏算了整个 encoder）

=== ToyVLA (float32) ===
batch  GFLOPs/iter  latency(ms)   TFLOPS     MFU  GPU busy
    1         5.59        3.591     1.56   24.4%       93%
    8        44.71       11.627     3.85   60.2%       96%
   32       178.84       40.060     4.46   69.9%       97%

=== ToyVLA (float16) ===
batch  GFLOPs/iter  latency(ms)   TFLOPS     MFU  GPU busy
    1         5.59        2.195     2.55    1.6%       54%
    8        44.71        2.776    16.11   10.3%       92%
   32       178.84        9.000    19.87   12.7%       96%
  128       715.38       41.703    17.15   10.9%       97%

=== TinyTransformer T=256 (float16) ===
batch  GFLOPs/iter  latency(ms)   TFLOPS     MFU  GPU busy
    1         6.98        0.698     9.99    6.4%       42%
    8        55.83        1.236    45.17   28.8%       96%
   64       446.68       11.308    39.50   25.2%       97%
```

解读：

1. **GPU busy 97%，MFU 只有 11%**（ToyVLA fp16，batch=128）。这就是"利用率指标只说明忙不忙"的直接证据。ToyVLA 里有大量卷积、BN、LayerNorm 等访存密集算子，它们让 GPU 一直有 kernel 在跑，但用不上 Tensor Core 的算力。
2. **FP32 的 MFU 反而更高**（70%），因为它的分母（6.39 TFLOPS）小得多。MFU 只能在同一个精度和峰值下比较。从绝对速度看，fp16 batch=32 用时 9.0 ms，fp32 是 40.1 ms，fp16 快 4.5 倍。
3. **batch=1 时 GPU busy 只有 42%~54%**。单次推理太短，CPU 下发 kernel 的速度跟不上，GPU 有一半时间在空等，这是典型的 Launch-bound，可以用 CUDA Graph 或 `torch.compile` 来解决（第 04、07 章）。
4. **FLOPs 统计工具会悄悄漏算**：推理模式下 `nn.TransformerEncoderLayer` 会走 fastpath 融合算子，`FlopCounterMode` 不认识它，于是记为 0；CPU 上的 SDPA 同样会被漏掉。统计 FLOPs 之前，先用经验公式粗算一遍做交叉验证。

## 0.4 Roofline 模型（实测）

Roofline 回答的问题是：**在这块硬件上，这个算子理论上最快能跑多少，现在离上限还有多远**。

```
可达性能 = min( 峰值算力, 峰值带宽 × 计算强度 )
计算强度 AI = FLOPs / 访存字节数   (FLOP/Byte)
```

- 屋顶（水平线）= 峰值算力；房檐（斜线）= 峰值带宽；拐点 AI = 峰值算力 / 峰值带宽。
- AI 小于拐点就是 memory-bound，优化方向是**减少访存**（融合、降低精度、减少中间结果）；AI 大于拐点就是 compute-bound，优化方向是**用上更强的算力**（Tensor Core、更低精度、更好的 kernel）。
- 两类优化在图上的移动方向：**算子融合**减少访存字节，点**向右**移；**实现优化**（合并访存、更好的 tiling）不改变 AI，点**向上**移。

### 实测峰值（[02_roofline.py](02_roofline.py)）

```
峰值带宽 (copy 256MB)        :    226.4 GB/s
FP32 理论峰值 (CUDA core)     :     8.06 TFLOPS  = 20 SM x 128 x 2 x 1.575 GHz
FP32 matmul 实测(最优尺寸)    :     6.39 TFLOPS  (79% of 理论)
TF32 matmul 实测(最优尺寸)    :    65.29 TFLOPS
FP16 matmul 实测(最优尺寸)    :   156.81 TFLOPS
BF16 matmul 实测(最优尺寸)    :   162.45 TFLOPS
FP16 持续满载 5s (8192)       :   104.13 TFLOPS  <- 功耗墙，见下文
拐点（FP16 屋顶 / 带宽）       :    692.5 FLOP/B  <- AI 低于它就是 memory-bound
拐点（FP32 屋顶 / 带宽）       :     28.2 FLOP/B
```

> 带宽在多次运行之间有 ±5% 的波动（226–236 GB/s），属于正常范围。

![Thor 实测 Roofline](outputs/roofline.png)

### 各算子在图上的位置

```
op                    AI(FLOP/B)        ms    GFLOPS     GB/s  判定
add fp32                    0.08     1.905      17.6    211.3  memory-bound, 达到屋顶 93%
relu fp32                   0.12     1.256      26.7    213.7  memory-bound, 达到屋顶 94%
a*b+c fp32                  0.12     2.433      27.6    220.7  memory-bound, 达到屋顶 97%
softmax fp16                1.25     1.280     131.1    104.9  memory-bound, 达到屋顶 46%
layernorm fp16              2.00     0.673     398.6    199.3  memory-bound, 达到屋顶 88%
GEMV (cold L2)              1.00     0.309     108.6    108.6  memory-bound, 达到屋顶 48%
GEMV (hot L2!)              1.00     0.036     929.6    930.0  memory-bound, 达到屋顶 411%
GEMM 256                   85.33     0.010    3339.4     39.1  memory-bound, 达到屋顶 17%
GEMM 512                  170.67     0.030    9098.3     53.3  memory-bound, 达到屋顶 24%
GEMM 1024                 341.33     0.099   21606.2     63.3  memory-bound, 达到屋顶 28%
GEMM 2048                 682.67     0.256   67083.7     98.3  memory-bound, 达到屋顶 43%
GEMM 4096                1365.33     1.050  130944.1     95.9  compute-bound, 达到屋顶 84%
GEMM 8192                2730.67    10.449  105229.5     38.5  compute-bound, 达到屋顶 67%
GEMM 512 fp32              85.33     0.100    2691.2     31.5  compute-bound, 达到屋顶 42%
GEMM 2048 fp32            341.33     3.000    5725.8     16.8  compute-bound, 达到屋顶 90%
GEMM 8192 fp32           1365.33   172.034    6391.2      4.7  compute-bound, 达到屋顶 100%
```

> 在单次运行内部，"memory/compute-bound"的判定依据是本次实测的拐点。GEMM 2048 的 AI（683）非常接近拐点（693），不同次运行之间的判定可能会翻转。

从实测数据中能读出这些结论：

1. **逐元素算子贴着带宽屋檐**（达到 93%~97%）：它们的 AI 只有 0.1 左右，性能完全由带宽决定。把 `add` 写得再精巧也不会更快，唯一的办法是**减少访存**，比如把多个逐元素算子融合成一个 kernel（第 07 章）。
2. **softmax 只达到屋顶的 46%**：一行 8192 个元素要先求 max、再求 sum、最后归一化，实现中会多次读取同一份数据，实际访存量大于理论最小值。这说明实现上还有优化空间（参考 online softmax 或 FlashAttention 的做法）。
3. **GEMV（batch=1 的 Linear，相当于 LLM decode）的 AI = 1**，即使在 Tensor Core 上也是严重的 memory-bound，这就是 LLM decode 慢的根本原因。prefill 阶段相当于大 GEMM，是 compute-bound；decode 阶段相当于 GEMV，是 memory-bound。所以 decode 阶段的主流优化手段是权重量化（减少字节数）和增大 batch（提高 AI）。
4. **小 GEMM 远离屋顶**：256 尺寸只达到 17%。数据量太小，切出来的 tile 不够分给 20 个 SM；kernel 启动和尾部开销的占比也高。
5. **FP16 Tensor Core 比 FP32 CUDA core 快 24 倍**（156.8 vs 6.4 TFLOPS），这是"开半精度"收益巨大的原因。但只有 compute-bound 的算子才能享受到这 24 倍，memory-bound 的算子开半精度最多快 2 倍（字节数减半）。

### ⚠️ 实测中踩到的两个坑

**坑 A：L2 缓存让带宽"超过物理极限"**。不刷 L2 时，GEMV 测出 930 GB/s，是 DRAM 峰值的 4 倍。原因是 4096×4096 的 fp16 权重正好 32 MB，等于 Thor 的 L2 容量，反复运行时权重一直留在 L2 里。实际推理时，L2 里会有其他层的权重和激活，这个结果没有参考意义。
→ **做 benchmark 前，要么让数据量远大于 L2，要么每次计时前写一块大缓冲区把 L2 刷掉**（见 `flush_l2()`）。

**坑 B：频率显示已锁定，持续满载时吞吐仍会掉 1/3**。8192×8192 的 FP16 GEMM 连续跑 5 秒，平均只有 104 TFLOPS，而单次突发能到 157 TFLOPS。下面是同时用 jtop 采样的结果（采样脚本见第 01 章）：

```
t(s) freqMHz load% VDD_GPU(mW) TOT(mW) tempC
(1.9, 1575, 97.0, 21217, 39873, 47.468)
(3.1, 1575, 98.0, 91194, 127886, 53.406)
(4.9, 1575, 98.0, 95735, 133398, 55.906)
(7.3, 1575, 97.0, 95735, 133497, 57.031)
```

jtop 报告的频率始终是 1575 MHz，但 GPU rail 功耗达到约 96 W，整机约 133 W，最可能的原因是撞上了功耗墙（硬件层面的降速，devfreq 接口上看不出来）。
→ **"峰值算力"要说清楚是突发峰值还是持续峰值**。本教程的 Roofline 屋顶使用突发峰值（单次迭代的最好成绩，很稳定）；评估长时间运行的服务时，应该使用持续吞吐。

## 0.5 瓶颈类型与诊断方法

| 类型 | 本质 | 诊断特征 | 验证工具 | 典型修复 | 本教程中的例子 |
|---|---|---|---|---|---|
| **CPU / Python bound** | Python 开销、预处理 | GPU 时间利用率低，CPU 单核跑满 | jtop、VizTracer、py-spy | 向量化、多进程、GPU 预处理 | `py_preprocess` 开关（第 02 章） |
| **同步 bound** | `.item()`、`.cpu()`、`cudaDeviceSynchronize` | 时间线上 CPU 阻塞在同步调用上 | torch.profiler（CPU 侧）、nsys | 去掉不必要的同步，结果统一取一次 | `sync_in_loop` 开关（第 03/04 章） |
| **Launch-bound** | kernel 启动往返延迟 | GPU busy < 50%，kernel 又多又碎 | nsys 看 kernel 之间的 gap | CUDA Graph、kernel 融合、`torch.compile` | `eager_small_ops` 开关、`ManySmallOps`（第 04 章） |
| **Memory-bound** | 带宽打满 | Memory BW > 70% Peak，AI < 拐点 | Roofline、ncu 看 Memory 吞吐 | 融合、低精度、数据排布 | 0.4 中的逐元素算子和 GEMV |
| **Register-bound** | 寄存器压力导致 Occupancy 低 | Occupancy < 40%，Regs/Thread > 64 | ncu 看 Occupancy | 精简变量、`-maxrregcount`、调 block size | 第 05 章 |
| **Control-flow-bound** | warp divergence | IPC < 1.5，分支效率低 | ncu 看 IPC / Branch | 减少分支、按 warp 对齐数据 | 第 05 章 |

两个关键结论：

- **CUDA Graph 只对 Launch-bound 有效**，对 Memory/Register/Control-flow-bound 无效（📎 CUDA 笔记·第三部分）。先诊断，再选手段。
- 诊断顺序遵循**漏斗模型**：先看系统级指标（第 01 章），确认 GPU 是否空闲；GPU 空闲就去 CPU 侧找原因（第 02/03 章）；GPU 忙碌但慢，就用时间线看分布（第 04 章）；最后定位到具体 kernel 再深挖（第 05 章）。

## 0.6 正确的计时方法

[01_timing_pitfalls.py](01_timing_pitfalls.py) 的实测输出：

```
=== 坑 1：首次调用（冷启动）===
第 1 次调用:   98.941 ms
第 2 次调用:   22.628 ms
第 3 次调用:   22.373 ms

=== 坑 2：不同步就计时 ===
不 sync:  median    0.010 ms  <- 只测到了 kernel launch 的 CPU 耗时
有 sync:  median   22.250 ms  <- 真实的 GPU 执行时间
连续 2000 次不 sync: 前 5 次 [0.129, 0.019, 0.013, 0.012, 0.012]
                     后 5 次 [22.194, 22.202, 22.251, 22.473, 22.2]  <- launch 队列满后 CPU 被阻塞

=== 坑 3：wall-clock vs CUDA Event ===
matmul wall                  p50=   22.266ms  p90=   22.317ms  p99=   22.583ms  ...
matmul event                 p50=   22.200ms  p90=   22.210ms  p99=   22.362ms  ...
=> FP32 matmul 实测算力 ≈ 6.19 TFLOPS

=== 坑 4：小 kernel 时 wall 比 event 多出 launch/同步开销 ===
tiny op wall                 p50=    0.015ms ...
tiny op event                p50=    0.013ms ...

=== 坑 5：只看均值会掩盖抖动 ===
cnn with hiccup              p50=    2.838ms  p90=    5.370ms  p99=   22.931ms  mean=    4.853ms ...
mean/p50 = 1.71，p99/p50 = 8.08  <- 尾延迟才是实时系统关心的
```

| 坑 | 现象 | 原因 | 正确做法 |
|---|---|---|---|
| 1 冷启动 | 第一次比后面慢 4 倍 | cuBLAS handle 初始化、选择算法、懒加载 kernel | 先 warmup 若干次，丢弃结果 |
| 2 不同步 | 测出 0.01 ms，比真实值快 2000 倍 | 只测到了 CPU 把 kernel 放进队列的时间 | 计时前后都要 `torch.cuda.synchronize()` |
| 2' 队列反压 | 连续不同步时，前几次很快，后面突然变慢 | launch 队列满了，CPU 被阻塞 | 同上。这个现象也说明 GPU 是瓶颈时 CPU 会被反压 |
| 3 wall vs event | 大 kernel 时两者基本一致 | | 纯 GPU 算子用 Event，端到端流水线用 wall |
| 4 小 kernel | wall 比 event 多出约 2 μs | wall 包含了 launch 和同步的开销 | 测微秒级的 kernel 用 Event，或者测 N 次循环的总时间 |
| 5 只看均值 | 均值 4.85 ms，p50 只有 2.84 ms | 偶发卡顿把均值拉高 | 报告 p50/p90/p99；实时系统重点看 p99 |
| 6 L2 缓存 | 带宽超过物理极限 | 数据留在 L2 里 | 数据量远大于 L2，或者先刷新 L2（0.4 坑 A） |
| 7 功耗墙 | 吞吐随运行时长下降 | 持续满载触发功耗墙 | 说清楚是突发值还是持续值（0.4 坑 B） |

CUDA Event 的原理（📎 CUDA 笔记·STREAM 和 EVENT）：`event.record()` 往 stream 里插入一个"打时间戳"的命令，GPU 执行到这里时记录下时间。所以两个 event 之间的差值，是 GPU 时间线上的时间，不受 CPU 侧开销干扰。但如果两个 event 之间 CPU 有阻塞，GPU 空等的时间也会被算进去。

## 0.7 共享工具 common/

后续各章都使用 [common/](../common/) 里的工具：

```python
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import benchmark, compare, region, print_env, ToyVLA, SlowPipeline, PipelineConfig

st = benchmark(fn, mode="event", warmup=10, iters=50)   # 返回 Stats：p50/p90/p99/mean/std/throughput
print(st.summary())
compare([st_baseline, st_optimized])                      # 打印加速比对比表

with region("preprocess"):   # 同时生成 NVTX range（nsys 可见）和 record_function（torch.profiler 可见）
    ...
```

| 文件 | 内容 |
|---|---|
| [bench_utils.py](../common/bench_utils.py) | `benchmark` / `compare` / `cuda_timer` / `region` |
| [env_info.py](../common/env_info.py) | 打印环境信息（版本、nvpmodel、频率、温度） |
| [models.py](../common/models.py) | `TinyCNN`、`TinyTransformer`、`ManySmallOps`、`ToyVLA` |
| [pipeline.py](../common/pipeline.py) | 带 6 个低效开关的慢流水线 `SlowPipeline` |
| [selftest.py](../common/selftest.py) | 自测，同时输出每个开关的影响（0.1 节的表） |

## 常见坑

1. 计时不同步、不做 warmup：见 0.6，这是最常见的错误。
2. 只看 GPU 利用率：利用率 97% 时，MFU 可能只有 11%（0.3）。
3. 不了解 L2 容量：Thor 的 L2 有 32 MB，小于 32 MB 的数据反复跑，测到的是 L2 带宽（0.4 坑 A）。
4. 不区分突发峰值和持续峰值：功耗墙会让持续吞吐比突发值低 1/3（0.4 坑 B）。
5. 完全信任 FLOPs 统计工具：fastpath 融合算子会被漏算（0.3）。
6. 凭直觉判断瓶颈：Thor 上 pinned memory 几乎没有收益（0.1）。

## 练习

见 [exercises.md](exercises.md)。

## 小结

- 先测量，再优化；用 Amdahl 定律判断一项优化值不值得做。
- "利用率"有多层含义，MFU 才反映实际产出的算力。
- Roofline 用于判断 memory-bound 还是 compute-bound，进而决定优化方向。Thor 实测：DRAM 约 230 GB/s，FP16 Tensor Core 约 157 TFLOPS（突发），拐点约 690 FLOP/B。
- 正确计时：warmup、同步、看分位数、刷 L2、注意功耗墙。
- 下一章：[第 01 章 系统级监控](../01_system_monitor/README.md)，学习用 jtop、nvidia-smi 做第一层的粗诊断。
