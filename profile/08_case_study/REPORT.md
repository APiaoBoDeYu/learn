# 性能分析报告：ToyVLA 慢流水线（Jetson AGX Thor）

> 按 [report_template.md](report_template.md) 填写，所有数字来自本机实测：
> [outputs/01_diagnose.log](outputs/01_diagnose.log)、[outputs/02_optimize.log](outputs/02_optimize.log)、[outputs/03_accuracy.log](outputs/03_accuracy.log)。
> 日期 2026-11-06；容器 `brain_thor_yhy`（privileged，镜像 v1.3.0_thor）。

## 1. 背景与目标

- 被测对象：`common/pipeline.py` 的 `SlowPipeline`（ToyVLA，41.4 M 参数）：相机帧 480×640 → 预处理 → 推理 → 后处理（动作缩放、裁剪、平滑），带 6 个"低效开关"。
- 场景：单请求、batch=1、低延迟（机器人控制回路）。
- 目标：端到端 p50 尽可能低；精度相对 fp32 基线的最大误差 < 2e-3（输出范围 [-1, 1]）。
- 约束：不改模型结构和权重。

## 2. 环境

| 项 | 值 |
|---|---|
| 硬件 | Jetson AGX Thor，GPU 20 SM（sm_110），126 GB 统一内存，14 核 CPU |
| 功耗模式 | MAXN + jetson_clocks，GPU 锁定 1575 MHz（jtop 读取；ncu 的 `--clock-control` 在 Thor 上不生效） |
| 软件 | 驱动 580.00，CUDA 13.0，cuDNN 9.17，torch 2.9.1，triton 3.8.0，TensorRT 10.14.1，nsys 2025.3.2，ncu 2025.3.1 |
| 权限 | privileged：ncu、GPU metrics、CPU 采样均可用；torch.profiler 拿不到 GPU 时间（平台限制） |
| 峰值参考 | DRAM 240 GB/s，FP32 6.39 TFLOPS，FP16 Tensor Core 155.6 TFLOPS（第 00 章实测） |

## 3. 基线（S0）

| 指标 | 值 | 测量方法 |
|---|---|---|
| 端到端延迟 | p50 **10.321 ms**，p99 10.846 ms | wall，200 步，warmup 20 步，不开 profiler（02_optimize） |
| 各阶段 | 预处理 2.997 / 推理 4.256 / 后处理 3.689 ms | 阶段边界同步后计时，60 步中位数 |
| GPU 利用率 | jtop GPU 负载 42%（02_optimize）/ 46%（01_diagnose）；nsys GPU busy 34.2% | jtop 3~4 s；nsys 20 步 |
| 功耗 / 能耗 | 34.2 W（GPU 15.4 W）→ **353 mJ/step** | jtop 3 s 平均 × p50 |
| MFU | 0.35% | 5.59 GFLOP（FlopCounterMode）/ p50 / FP16 峰值 |

## 4. 诊断

| 层 | 工具 | 关键证据 | 结论 |
|---|---|---|---|
| L0 | jtop | GPU 负载 46%，最忙的 CPU 核 78%，GPU 功耗 15.4 W | GPU 没吃满，CPU 侧有瓶颈 → 先看 L1 |
| L1 | cProfile（50 步） | `postprocess` 21.8%、`_preprocess_python` 15.0%、`_transformer_encoder_layer_fwd` 6.0% | 后处理和 Python 预处理是 CPU 热点 |
| L2 | torch.profiler（CPU） | 每步 2359 次 aten 调用：`aten::mul` 231 次、`aten::select` 491 次、`aten::item` 14 次 | 后处理由大量小算子组成；每步 14 次同步 |
| L3 | nsys + SQL（20 步） | GPU busy 34.2%；postprocess 每步 397 个 kernel，98% 短于 5 μs；infer 120 个 kernel、GPU 3.158 ms | 后处理是 Launch-bound；推理占 GPU 时间的大头 |
| L4 | ncu（第 05 章 5.9） | 推理中 fp32 GEMM `Kernel2` 占 47.7%（Occupancy 16.7%、Tail Effect），`nchwToNhwcKernel` 8.5% | 推理用 fp16 + channels_last；进一步可以交给 TensorRT |

瓶颈清单（按预期收益排序；"预期"来自 `common/selftest.py` 中逐个关闭开关的结果）：

| # | 瓶颈 | 类型 | 证据 | 预期收益 |
|---|---|---|---|---|
| 1 | 后处理的小算子 | Launch-bound | L1 21.8%、L3 397 kernel/step | 约 1.4x（selftest：eager_small_ops 1.37x） |
| 2 | Python 预处理 | CPU-bound | L1 15.0% | 约 1.3x（selftest：py_preprocess 1.29x） |
| 3 | 循环中的 `.item()` | 同步 | L2 14 次/step | 约 1.1~1.2x |
| 4 | fp32 推理 | Compute（GEMM Occupancy 低） | L4 47.7% | 推理部分约 1.5~2x |
| 5 | pageable 拷贝、反复分配 | 凭直觉的"瓶颈" | 没有直接证据 | 约 1.0x（selftest：no_pin 0.99x、realloc 1.02x） |

## 5. 优化记录

| 步骤 | 改动 | p50 / p99 (ms) | 本步 | 累计 | 预处理 / 推理 / 后处理 (ms) | 功耗 W | mJ/step | MFU | 误差 vs S0 |
|---|---|---|---|---|---|---|---|---|---|
| S0 | 基线 | 10.321 / 10.846 | 1.00x | 1.00x | 2.997 / 4.256 / 3.689 | 34.2 | 353 | 0.35% | 0 |
| S1 | 后处理向量化 | 7.503 / 8.276 | 1.38x | 1.38x | 2.912 / 4.313 / 0.506 | 38.8 | 291 | 0.48% | 0 |
| S2 | 预处理：只拷贝 uint8，归一化放到 GPU | 4.935 / 5.497 | 1.52x | 2.09x | 0.699 / 4.131 / 0.459 | 46.1 | 228 | 0.73% | 1.2e-4 |
| S3 | 去掉循环中的 `.item()` | 4.641 / 5.198 | 1.06x | 2.22x | 0.706 / 4.105 / 0.154 | 47.7 | 222 | 0.77% | 1.2e-4 |
| S4 | fp16 | 3.277 / 3.463 | 1.42x | 3.15x | 0.695 / 2.385 / 0.144 | 36.9 | 121 | 1.10% | 8.7e-4 |
| S5 | pinned + 预分配 | 3.341 / 5.386 | 0.98x | 3.09x | 0.799 / 2.310 / 0.144 | 33.1 | 111 | 1.08% | 8.7e-4 |
| S6 | channels_last | 3.271 / 4.444 | 1.02x | 3.16x | 0.916 / 2.326 / 0.148 | 32.3 | 106 | 1.10% | 8.7e-4 |
| S7 | torch.compile reduce-overhead | 1.819 / 4.311 | 1.80x | 5.67x | 1.162 / 0.821 / 0.139 | 38.0 | 69 | 1.98% | 7.7e-4 |
| S8 | TensorRT fp16 + CUDA Graph（替代 S7） | 1.207 / 1.719 | 1.51x | 8.55x | 0.772 / 0.389 / 0.119 | 34.9 | 42 | 2.98% | 1.2e-3 |
| S9 | resize 也放到 GPU | **0.747** / 1.210 | 1.62x | **13.82x** | 0.274 / 0.384 / 0.103 | 42.8 | **32** | **4.81%** | 9.1e-4 |

![stages](outputs/stages.png)

说明：
- "本步"是相对上一行的加速比；S8 的改动是用 TensorRT 替换 S6 的 PyTorch 模型，与 S7 的 compile 不叠加，所以 S8 的"本步"是相对 S7 的结果。
- 各阶段时间是在阶段边界同步后分别计时的，用来看时间花在哪里，总和会略大于端到端 p50。
- 功耗：GPU 越忙，平均功耗越高（S2/S3 时 GPU 负载升到 70%+，总功耗 46~48 W），但每步能耗一直在下降：**最终 353 → 32 mJ/step（11 倍）**。

无效 / 负优化：

| 改动 | 预期 | 实测 | 原因 |
|---|---|---|---|
| S5 pinned + 预分配 | 凭直觉"拷贝和分配很慢" | 0.98x（单独交替重复 3 次：S4 p50 3.17~3.18 ms，S5 3.19~3.20 ms，没有差别） | Thor 是统一内存，pageable 拷贝不慢；caching allocator 让分配本身很便宜（第 03、07 章）。诊断中本来就没有证据指向它，**这一步不该做**，保留下来作为反例 |
| S6 channels_last | 第 07 章 TinyCNN fp16 上有 1.56x | 1.02x | ToyVLA 的推理时间主要在 Transformer 部分（第 05 章 5.9），卷积占比小 |

## 6. 结果与剩余瓶颈

- 端到端 p50 从 10.321 ms 降到 **0.747 ms（13.8 倍）**，p99 从 10.846 ms 降到 1.210 ms；每步能耗降低 11 倍；最大误差 9.1e-4，满足目标。
- 剩余时间：预处理 0.274 + 推理 0.384 + 后处理 0.103 ms。推理部分已经接近第 06 章 TensorRT fp16 + CUDA Graph 的 GPU 时间（0.345 ms）。
- nsys（`--cuda-graph-trace=node`）：最终 GPU busy 37.5%，每步 84 个 kernel（基线 517 个）。**GPU 仍有约 60% 的时间是空闲的**，因为预处理、推理、后处理串行执行，CPU 侧的 Python 调度和拷贝等待夹在 GPU 工作之间。
- 下一步（未实测）：(1) 把预处理 + 推理 + 后处理整体捕获成一个 CUDA Graph；(2) 用两个流水级让第 i+1 帧的预处理与第 i 帧的推理重叠（注意 Thor 上拷贝由 SM 执行，第 04 章 4.8）；(3) 量化（需要校准数据，第 06 章 INT8 未校准时误差 0.83）。

## 7. 风险与未解决问题

- **观察者效应**：nsys 下慢版的 pageable 拷贝偶尔需要 1~6 ms，不开 nsys 时不存在（第 04 章 4.8 D）；cProfile 会让程序慢约 40%。本报告的延迟数字全部来自不开 profiler 的运行。
- **nsys 与 CUDA Graph**：不加 `--cuda-graph-trace=node` 时，CUDA Graph 内的 kernel 不出现在 KERNEL 表中，最终阶段的 GPU busy 会被算成 6.1%（实际是 37.5%）。本报告在第一次计算时就犯了这个错误。
- **尾延迟**：S5~S8 的 p99 波动很大：本次运行中比 p50 高出 2.0 / 1.2 / 2.5 / 0.5 ms（S5/S6/S7/S8），前一次运行中 S8 的 p99 是 3.70 ms。对 S8 逐阶段计时发现，尾延迟来自预处理中 147 KB 的 H2D 拷贝（600 步中有 7 次超过 1 ms，最长 3.4 ms）。单独测试 pinned 拷贝：147 KB 的 p99 是 236 μs、最长 2.7 ms，而 256 KB / 900 KB 的 p99 只有 46 / 54 μs。147 KB 小于约 256 KB 的阈值，会走 Copy Engine（第 04 章 4.8 C），**推测**尾延迟与此有关（机制未验证）。S9 改成拷贝 900 KB 的原始帧后，p99 降到 1.2 ms。把 OpenCV 线程数设为 1 对尾延迟没有影响（已排除）。
- **S7 中预处理变慢**（0.916 → 1.162 ms），原因未深究。
- **平台限制**：torch.profiler 拿不到 GPU 时间；ncu 没有 DRAM 指标，带宽是用 L2 计数换算的（第 05 章 5.3）。
- **TensorRT engine 与平台绑定**：sm_110 + TensorRT 10.14 构建，换平台需要重新构建；INT8 需要校准后才能用。

## 附录：复现

```bash
/opt/venv/bin/python 06_tensorrt/01_export_onnx.py && /opt/venv/bin/python 06_tensorrt/02_trtexec.py   # 生成 S8/S9 用的 engine
/opt/venv/bin/python 08_case_study/01_diagnose.py     # L0~L3 诊断 → outputs/01_diagnose.log
/opt/venv/bin/python 08_case_study/02_optimize.py     # S0~S9 → outputs/02_optimize.{log,json}、stages.png
/opt/venv/bin/python 08_case_study/03_accuracy.py     # 精度 → outputs/03_accuracy.log
```

⚠️ `outputs/` 中的 nsys 报告（`s0_nsys.*`、`stage*_nsys.*`）包含采集时的完整环境变量，分享前需要检查（[plan/env_report.md](../plan/env_report.md)）。
