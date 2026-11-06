# 第 01 章 系统级监控：jtop / tegrastats / nvidia-smi

> 📎 对应笔记：[doc/程序性能分析工具.xmind](../doc/程序性能分析工具.xmind) →「jtop」（各页签截图与字段说明）、「nvidia-smi dmon -s u」
>
> 漏斗模型的第 0 层（L0）：**先用最便宜的工具，判断问题大概出在 CPU 还是 GPU**。

## 学习目标

1. 会用 jtop 的界面和 Python API，知道每个字段在 Thor 上是否可信；
2. 会用 nvidia-smi dmon / NVML，并清楚它们在 Jetson 上的限制；
3. 只看系统级指标，就能把一个负载粗分为 CPU-bound、Launch-bound、Memory-bound 或 Compute-bound；
4. 知道系统级指标的盲区：采样延迟、"利用率"的真实含义、功耗墙。

## 目录

- [1.1 工具一览](#11-工具一览)
- [1.2 jtop：界面与 Python API](#12-jtop界面与-python-api)
- [1.3 tegrastats](#13-tegrastats)
- [1.4 nvidia-smi / NVML](#14-nvidia-smi--nvml)
- [1.5 实战：用系统指标区分 5 种负载](#15-实战用系统指标区分-5-种负载)
- [1.6 功耗模式与锁频](#16-功耗模式与锁频)
- [1.7 判读流程](#17-判读流程)
- [常见坑](#常见坑) · [练习](exercises.md) · [小结](#小结)

| Demo | 内容 |
|---|---|
| [01_jtop_snapshot.py](01_jtop_snapshot.py) | jtop API 读取所有页签的信息 |
| [02_nvidia_smi.py](02_nvidia_smi.py) | NVML 字段在本机是否可用；负载运行期间 dmon 的输出 |
| [03_load_patterns.py](03_load_patterns.py) | **核心**：5 种负载的系统指标时间线 |
| [04_tegrastats_parse.py](04_tegrastats_parse.py) | 解析 tegrastats 日志 |
| [05_power_modes.py](05_power_modes.py) | 功耗模式对延迟的影响（默认只读） |
| [common/monitor.py](../common/monitor.py) | 共享组件：`JtopSampler` / `NvmlSampler` 后台采样器 |

---

## 1.1 工具一览

| 工具 | 平台 | 优点 | 局限 |
|---|---|---|---|
| **jtop**（jetson-stats） | Jetson | 信息最全（CPU 各核、GPU、EMC、功耗、温度、引擎），有 Python API | 采样有约 1 s 延迟；只有时间维度的利用率 |
| **tegrastats** | Jetson | 系统自带，可以写日志文件 | 需要 root；容器内通常没有；格式随版本变化 |
| **nvidia-smi / NVML** | 全平台 | 标准接口，x86 和 Jetson 通用 | **在 Jetson 上大部分字段不支持**（见 1.4） |
| nvtop | 全平台 | 好看的 TUI | 信息和 nvidia-smi 一样 |
| DCGM（`dcgmi dmon`） | 数据中心 GPU | 有 SM_ACTIVE、SM_OCCUPANCY、TENSOR_ACTIVE 等空间维度指标，开销低，可以用在生产环境 | Jetson 不支持 |

> 系统级工具的共同点：**只能告诉你"忙不忙"，不能告诉你"为什么慢"**。它们的作用是为下一步选对工具。

## 1.2 jtop：界面与 Python API

### 界面

终端运行 `jtop` 即可打开界面，各页签的截图和字段含义见 📎 xmind 笔记。重点字段：

| 页签 | 字段 | 性能分析中的用途 |
|---|---|---|
| ALL | GPU / CPU 各核 / EMC / 功耗 | 一眼判断谁在忙 |
| GPU | load、freq、governor | 确认频率是否已锁定（governor=performance，min=max） |
| CPU | 每核负载、频率 | Python 单线程瓶颈表现为**某一个核** 100% |
| MEM | RAM、EMC 频率 | Jetson 是统一内存，GPU 显存也计入 RAM |
| ENG | 硬件引擎（PVA、VIC、NVENC/NVDEC 等） | 确认视频编解码、PVA 是否真的在工作 |
| CTRL | nvpmodel、jetson_clocks、风扇 | benchmark 之前先检查这一页 |
| INFO | JetPack/L4T/CUDA/cuDNN/TRT 版本 | 记录实验环境 |

### Python API（[01_jtop_snapshot.py](01_jtop_snapshot.py)）

```python
from jtop import jtop
with jtop() as j:
    j.nvpmodel, j.jetson_clocks       # 功耗模式、锁频状态
    j.gpu                             # {'Thor': {'status': {'load': ..}, 'freq': {'cur','min','max','governor'}}}
    j.cpu['cpu'][i]                   # 每个核：idle/user/system、频率、governor
    j.memory['RAM'], j.memory['EMC']
    j.power['rail'], j.power['tot']   # 各路功耗（mW）
    j.temperature, j.engine, j.processes
    j.stats                           # 扁平化的一份快照，适合定期采样
```

本机输出（节选）：

```
Model    : NVIDIA Jetson AGX Thor Developer Kit
L4T      : 38.4.0   JetPack: 7.2
Libraries: {'CUDA': '13.0.88', 'cuDNN': '9.17.0', 'TensorRT': '10.14.1.48'}
nvpmodel      : MAXN    (可选模式: ['MAXN', '120W', '90W', '70W'])
jetson_clocks : True
Thor: load=0.0%  freq=1575 MHz (min 1575 / max 1575)  governor=performance
14 cores, governor=performance, freq=2601 MHz
RAM  used 13.5 / 120.7 GB  (cached 42.3 GB)  ← Jetson 统一内存：CPU 和 GPU 共用这一份
EMC  freq 4266 MHz (max 4266)  load=0%
VDD_CPU_SOC_MSS     10.99 W
VDD_GPU              4.17 W
VIN_SYS_5V0          6.36 W
TOTAL               21.52 W
=== ENG 页：硬件加速引擎 ===
APE / PVA0_CPU_AXI / PVA0_VPS / SE / VIC        OFF
```

两个和 Orin 不同的地方：**Thor 没有 DLA**（ENG 页里只有 PVA、VIC、SE、APE），所以 TensorRT 的 DLA offload 在 Thor 上不可用；**EMC load 始终为 0**（见 1.5）。

### 后台采样：`common/monitor.py`

想把系统指标和推理过程对齐，就要在推理运行时于后台线程采样：

```python
from common.monitor import JtopSampler
with JtopSampler(interval=0.1) as s:
    s.mark("warmup");  run_warmup()
    s.mark("infer");   run_inference()
print(s.summary(["gpu_load", "cpu_max", "p_gpu_w"]))   # 按阶段求平均
s.to_csv("outputs/monitor.csv")
```

采样字段：`gpu_load`、`gpu_freq_mhz`、`emc_load`、`cpu_avg`、`cpu_max`（最忙那个核）、`ram_gb`、`temp_gpu`、`p_gpu_w`、`p_cpu_soc_w`、`p_tot_w`。

## 1.3 tegrastats

```bash
sudo tegrastats --interval 100 --logfile /tmp/tegra.log   # 后台采样 100 ms 一次，写入文件
sudo tegrastats --stop
```

容器内没有这个命令。下面是在宿主机上采集的真实输出（[outputs/tegrastats_host_sample.log](outputs/tegrastats_host_sample.log)）：

```
11-05-2026 21:30:32 RAM 13114/123597MB (lfb 14x4MB) CPU [0%@2601,...,11%@2601,...] EMC_FREQ 0%@4266
GR3D_FREQ @[1574,1574,1574] NVENC0_FREQ @1691 ... VIC off OFA_FREQ @1691 PVA0_FREQ off APE 300
cpu@37.093C tj@38.75C soc012@36.812C gpu@38.75C soc345@36.593C
VDD_GPU 4927mW/4927mW VDD_CPU_SOC_MSS 7580mW/7580mW VIN_SYS_5V0 5374mW/5374mW
```

| 字段 | 含义 |
|---|---|
| `CPU [x%@MHz,...]` | 每个核的负载和频率 |
| `EMC_FREQ x%@MHz` | 内存控制器负载和频率（Thor 上负载显示为 0） |
| `GR3D_FREQ @[1574,1574,1574]` | GPU 每个 GPC 的频率。**Thor 上没有负载百分比**（Orin 上的格式是 `GR3D_FREQ 45%@1300`） |
| `VDD_GPU 4927mW/4927mW` | 当前功耗 / 平均功耗 |

[04_tegrastats_parse.py](04_tegrastats_parse.py) 把这些字段解析成字典，便于和推理日志按时间戳对齐：

```
11-05-2026 21:30:32  RAM 13114/123597MB  CPU max 11% avg 0.8% @2601MHz  EMC 0%@4266  GPU load=None freq=[1574, 1574, 1574]  gpu 38.75C  VDD_GPU 4.9W(avg 4.9) ...
```

## 1.4 nvidia-smi / NVML

```bash
nvidia-smi dmon -s u            # 每秒一行：sm% mem% enc% dec% jpg% ofa%
nvidia-smi dmon -s pucm -d 1    # p=功耗/温度 u=利用率 c=频率 m=显存
nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv -lms 100
```

Python 里用 `pynvml`（NVML 就是 nvidia-smi 底层用的库）。[02_nvidia_smi.py](02_nvidia_smi.py) 探测了常用接口在 Thor 上是否可用：

```
NVML query                   result
utilization (gpu/mem %)      ✅ gpu=0% mem=0%
memory info                  ❌ Not Supported
temperature                  ✅ 37
power usage (mW)             ✅ 4169
SM clock (MHz)               ❌ Not Supported
mem clock (MHz)              ❌ Not Supported
throttle reasons             ❌ Not Supported
util samples (history)       ❌ Not Supported
running processes            ✅ 0         ← 实际有进程在用 GPU，但列表是空的
PCIe throughput              ❌ Not Supported
```

在负载运行期间执行 `dmon -s u`（负载依次为：空闲 2 s → 拷贝 4 s → GEMM 4 s）：

```
# gpu     sm    mem    enc    dec    jpg    ofa
    0      0      0      0      0      0      0
    0      0      0      0      0      0      0
    0      0      0      0      0      0      0
    0     98      0      0      0      0      0     ← 拷贝阶段（纯带宽负载），mem 列却是 0
    ...
    0     98      0      0      0      0      0     ← GEMM 阶段，sm 和拷贝阶段一样都是 98%
阶段起始时间(s): [(0.0, 'idle'), (2.0, 'copy'), (8.3, 'gemm')]
```

两个发现：

1. **在 Thor 上 nvidia-smi 只有 `sm%` 能用**，`mem` 列（显存带宽忙碌比例）始终为 0。NVML 也读不到频率和限频原因，所以无法通过 NVML 确认功耗墙降速。在 Jetson 上优先使用 jtop。
2. **拷贝阶段原定 4 s，实际用了 6.3 s**。CPU 以远快于 GPU 的速度往 stream 队列里塞 copy kernel，循环结束时队列里还积压了约 2 s 的任务，最后的 `synchronize()` 要等它们全部跑完。这说明 CUDA 队列的深度很可观，也再次说明计时必须同步。

## 1.5 实战：用系统指标区分 5 种负载

[03_load_patterns.py](03_load_patterns.py) 依次运行 5 种负载，用 `JtopSampler` 每 0.1 s 采样一次：

| 阶段 | 负载 | 构造方式 |
|---|---|---|
| 2 CPU-bound | Python 逐行做预处理 | `SlowPipeline(PipelineConfig.fast(py_preprocess=True))` |
| 3 launch-bound | 200 个极小的逐元素算子 | `ManySmallOps(200)`，输入只有 1024 个元素 |
| 4 memory-bound | 256 MB 拷贝 | `y.copy_(x)` |
| 5 compute-bound | FP16 GEMM 8192 | `a @ a` |

![5 种负载的系统指标](outputs/load_patterns.png)

```
phase                     gpu_load     cpu_max     cpu_avg    emc_load     p_gpu_w p_cpu_soc_w     p_tot_w    temp_gpu
1 idle                         0.0        41.8         5.9         0.0         9.2         9.0        24.4        38.6
2 CPU-bound                   17.9        77.6         9.4         0.0         9.9         9.7        25.9        38.7
3 launch-bound                32.8       100.0         9.7         0.0         8.8         9.7        24.3        38.9
4 memory-bound                86.8        82.2         7.7         0.0        15.3        17.6        47.0        40.3
5 compute-bound               97.5        13.5         2.1         0.0        88.6        19.3       124.7        56.1
6 cool-down                   29.1        13.2         1.9         0.0        45.5        12.7        68.6        48.5

GEMM 吞吐：前 3 次 118.9 TFLOPS → 最后 5 次 111.3 TFLOPS
```

> 表中的平均值受 jtop 约 1 s 采样延迟的影响（见下文第 1 条），阶段边界附近的数据会混入相邻阶段，所以看时间线图比看平均值更准确。

### 判读

| 负载 | GPU load | 最忙的 CPU 核 | GPU 功耗 | 判读依据 |
|---|---|---|---|---|
| CPU-bound | **低**（约 20%） | **100%** | 很低 | GPU 在等 CPU，CPU 只有一个核跑满（Python GIL） |
| Launch-bound | 中低（约 35%） | **100%** | 很低 | 和 CPU-bound 很像！需要用 nsys 看 kernel 之间的空隙才能区分（第 04 章） |
| Memory-bound | **约 98%** | 100%（自旋等待） | **中**（约 16 W） | GPU 很忙但功耗不高，说明在搬数据而不是在计算 |
| Compute-bound | **约 98%** | 低 | **高**（约 100 W） | GPU 很忙，功耗也高，Tensor Core 在全力计算 |

从时间线图里可以读出 4 个容易踩坑的结论：

1. **jtop 的数据比实际负载晚约 1 s**。CPU-bound 阶段从 t=2 s 开始，GPU load 到 t=3 s 才上升；compute 阶段在 t=22.3 s 结束，功耗到 t=23 s 才下降。用 jtop 数据对齐推理日志时，要把这个延迟考虑进去；时长小于 1 s 的事件，jtop 基本看不出来。
2. **"CPU 单核 100%"不代表 CPU-bound**。launch-bound 和 memory-bound 阶段，最忙的核也是 100%：GPU 处理不过来，队列满了以后，CPU 就在 `cudaLaunchKernel` 里**自旋**等待空位。而在 memory-bound 阶段的末尾（t≈14~16 s），CPU 在 `torch.cuda.synchronize()` 里**阻塞**等待，负载反而降到约 12%。
3. **GPU load 区分不了 memory-bound 和 compute-bound**，两者都约 98%。这正是第 00 章说的"时间维度的利用率只说明有没有 kernel 在跑"。**GPU 功耗能区分**：搬数据约 16 W，算矩阵约 100 W。在 Thor 上 EMC load 和 `dmon mem` 都读不到，功耗是唯一可用的辅助信号。
4. **功耗墙**：GEMM 开始约 1 s 后，GPU 功耗升到约 100 W，吞吐从 119 降到 111 TFLOPS，而 jtop 显示的频率始终是 1575 MHz（第 00 章 0.4 坑 B）。

## 1.6 功耗模式与锁频

| 命令（宿主机） | 作用 |
|---|---|
| `sudo nvpmodel -q` | 查询当前功耗模式 |
| `sudo nvpmodel -m <id>` | 切换功耗模式（Thor：MAXN / 120W / 90W / 70W） |
| `sudo jetson_clocks` | 把 CPU/GPU/EMC 锁在当前模式的最高频率 |
| `sudo jetson_clocks --show` | 查看当前频率 |

容器内没有这些命令，可以通过 jtop API 读取，也可以切换 nvpmodel（`j.nvpmodel = "120W"`）。⚠️ 但 **jtop 不一定能开启 jetson_clocks**：本机的 jetson_clocks 是在宿主机上用命令开启的，jtop 中的状态是 `uncontrolled`，实测 `j.jetson_clocks = True` / `set_enable(True)` 不报错也不生效（见下文实测）。

**benchmark 之前必须确认**：nvpmodel 模式、jetson_clocks 是否开启、GPU 温度，并记录在结果里（`common.print_env()` 会自动打印）。没有锁频时，DVFS 会根据负载动态调频，第一次迭代的频率可能很低，延迟的波动也会变大。

[05_power_modes.py](05_power_modes.py) 默认只读，只测量当前模式：

```
当前模式: MAXN  jetson_clocks=True  可选: ['MAXN', '120W', '90W', '70W']
[MAXN] fn  p50=    2.216ms  p90=    2.270ms  p99=    2.369ms  ...
```

加上 `--apply` 会依次切换各模式测量，结束后恢复原模式。⚠️ 这会修改**宿主机的全局设置**，影响机器上所有的程序，要在确认不影响其他服务后再运行。

### 实测：`--apply`（2026-11-06，经用户同意后运行）

日志 [outputs/05_power_modes_apply.log](outputs/05_power_modes_apply.log)：

```
当前模式: MAXN  jetson_clocks=True  GPU 1575 MHz（min 1575 / max 1575，performance）
可选模式（是否无需重启）: {'MAXN': True, '120W': True, '90W': True, '70W': True}
[MAXN] p50=2.214ms
切换到 120W ...  [120W] jetson_clocks=False  GPU 1386 MHz（min 315 / max 1386）   p50=2.169ms
切换到 90W ...   [120W] jetson_clocks=False  ...（jtop 仍显示 120W）            p50=2.210ms
切换到 70W ...   [120W] jetson_clocks=False  ...（jtop 仍显示 120W）            p50=2.182ms
已恢复：MAXN  jetson_clocks=False  GPU 1575 MHz（min 315 / max 1575）
```

1. **切换 nvpmodel 会关掉 jetson_clocks**：切到 120W 后 GPU 不再锁频（315~1386 MHz）。恢复到 MAXN 后，jetson_clocks 仍然是关闭的，GPU、CPU、EMC 都变回动态调频，而 jtop 又无法重新开启它（状态为 `uncontrolled`）。最后是在**宿主机上执行 `sudo jetson_clocks`** 才恢复的：`jetson_clocks --show` 显示 CPU 0~13 min = max = 2601 MHz，GPU（gpc）1575 MHz，EMC 4266 MHz，与运行前一致；恢复后 ToyVLA 的 p50 是 2.231 ms（运行前 2.216 ms）。`/var/lib/nvpmodel/status` 为 `pmode:0000`（MAXN），没有留下待重启后生效的变更。
2. **90W、70W 没有切换成功**：jtop 仍然显示 120W。运行结束后再查询，jtop 报告这两个模式需要重启才能生效（status 变成了 False，而运行前是 True），所以这两行数据**实际测的是 120W**，不能作为 90W/70W 的结论。（从 120W 切换时需要重启的原因未深究；猜测与这两个模式改变了 CPU 在线核数等需要重启的参数有关。）
3. **120W vs MAXN 几乎没有差别**（2.169 vs 2.214 ms）：ToyVLA fp16 batch=1 是 Launch-bound（第 04 章），GPU 功耗只有十几瓦，远没有达到 120W 模式的功耗上限，最高频率从 1575 降到 1386 MHz 对它的影响也很小。要看到功耗模式的影响，需要用持续满载的 compute-bound 负载（例如第 00 章的 GEMM 8192 持续 5 秒），而且要在每种模式下重新开启 jetson_clocks，保证比较条件一致。

**教训**：脚本只在结束时恢复了 nvpmodel，没有恢复 jetson_clocks（已修改：现在会保存并尝试恢复 jetson_clocks、打印恢复前后的频率；但 jetson_clocks 由宿主机命令开启时，仍需要在宿主机上手动执行 `sudo jetson_clocks`）。修改全局状态的脚本，**恢复之后一定要验证**（`jetson_clocks --show` 中 min = max），而不是只看函数有没有报错。

## 1.7 判读流程

```
                    ┌─────────────────────────┐
                    │ GPU load（jtop/dmon）    │
                    └────────────┬────────────┘
              低 (< 50%)         │          高 (> 90%)
        ┌────────────────────────┴───────────────────────┐
        ▼                                                ▼
  看最忙的 CPU 核                                     看 GPU 功耗
   ├─ 100% → CPU-bound 或 Launch-bound                 ├─ 接近功耗上限 → 偏 compute-bound
   │        → 第 02 章 (VizTracer/py-spy) 看 Python 热点  │   → 第 05 章 ncu 看 SOL / Tensor Core
   │        → 第 04 章 (nsys) 看 kernel 之间的空隙        └─ 偏低 → 偏 memory-bound
   └─ 不高 → I/O、锁、sleep、等待外部输入                    → 第 00 章 Roofline + 第 05 章 ncu
            → 第 02 章 py-spy 看线程在哪里阻塞
```

## 常见坑

1. **不看采样延迟**：jtop 约 1 s，`nvidia-smi dmon` 默认 1 s。小于 1 s 的事件基本看不见，需要用 nsys 这类 trace 工具。
2. **把 CPU 100% 当成 CPU-bound**：GPU 太忙时，CPU 会在 launch 调用里自旋（1.5 第 2 条）。
3. **把 GPU load 当成算力利用率**：memory-bound 和 compute-bound 的 GPU load 都约 98%（1.5 第 3 条）。
4. **在 Jetson 上相信 nvidia-smi 的 mem% 和 EMC load**：Thor 上它们始终为 0。
5. **benchmark 不记录功耗模式和锁频状态**：结果无法复现。
6. **以为锁频后频率就固定了**：功耗墙会在频率读数不变的情况下降速（1.5 第 4 条）。
7. **切换 nvpmodel 后不检查 jetson_clocks**：切换会关掉锁频，恢复模式时锁频不会自动恢复，jtop 也不一定能重新开启（1.6 实测）。

## 练习

见 [exercises.md](exercises.md)。

## 小结

- 系统级监控回答"谁在忙"，用于为下一步选对工具，成本最低。
- 在 Thor 上：jtop 是主力工具；nvidia-smi 只有 `sm%` 能用；EMC load 读不到；tegrastats 只能在宿主机上运行；Thor 没有 DLA。
- 判读要点：GPU load 低时看 CPU 核，GPU load 高时看功耗；要考虑约 1 s 的采样延迟。
- CPU-bound 和 Launch-bound 只看系统指标区分不开，下一章用 Python profiler 看 CPU 侧，[第 04 章](../04_nsight_systems/README.md)用 nsys 看 GPU 时间线。
- 下一章：[第 02 章 Python 级 Profiling](../02_python_profilers/README.md)。
