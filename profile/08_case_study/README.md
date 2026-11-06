# 第 08 章 综合实战：ToyVLA 慢流水线端到端优化

> 把第 00~07 章的工具和手段串起来：先按漏斗模型诊断，再按诊断结论逐项修复，每一步都测量、记录，最后写成报告。
> **结果**：端到端 p50 **10.321 → 0.747 ms（13.8 倍）**，每步能耗 353 → 32 mJ，MFU 0.35% → 4.81%，最大误差 9.1e-4。
> 完整报告：[REPORT.md](REPORT.md)；报告模板：[report_template.md](report_template.md)。
>
> 📎 对应笔记：[doc/CUDA.xmind](../doc/CUDA.xmind) →「性能分析·四大瓶颈」（诊断时的分类）；[doc/TensorRT_副本.xmind](../doc/TensorRT_副本.xmind) →「TRT Profiler 和 Nsight 如何配合使用」（S8 前的三步法）；[doc/程序性能分析工具.xmind](../doc/程序性能分析工具.xmind) →「jtop」「viztracer」「Nsight」（L0~L3 用到的工具）。

## 学习目标

1. 用漏斗模型（L0 → L4）在一个真实的流水线上定位瓶颈，并让诊断结论决定修复顺序；
2. 每一步修复都和上一步对比，同时记录延迟、尾延迟、功耗/能耗、MFU 和精度；
3. 识别"凭直觉的瓶颈"、观察者效应和测量错误，并在报告中如实记录。

## 目录

- [8.1 诊断](#81-诊断)
- [8.2 逐项修复](#82-逐项修复)
- [8.3 测量中踩到的坑](#83-测量中踩到的坑)
- [8.4 报告](#84-报告)
- [练习](exercises.md) · [小结](#小结)

| 文件 | 内容 |
|---|---|
| [case.py](case.py) | 10 个累积式的优化阶段（S0~S9）+ 测量函数（端到端、各阶段、jtop 功耗） |
| [01_diagnose.py](01_diagnose.py) | L0 jtop → L1 cProfile → L2 torch.profiler → L3 nsys + SQL；L4 引用第 05 章 |
| [02_optimize.py](02_optimize.py) | S0~S9 的测量、nsys GPU busy、`outputs/stages.png` |
| [03_accuracy.py](03_accuracy.py) | 每个阶段与 S0 的输出误差 |
| [report_template.md](report_template.md) / [REPORT.md](REPORT.md) | 报告模板 / 本次的实测报告 |

---

## 8.1 诊断

[01_diagnose.py](01_diagnose.py)，日志 [outputs/01_diagnose.log](outputs/01_diagnose.log)。每一层只回答一个问题，然后决定看哪一层：

| 层 | 问题 | 证据 | 下一步 |
|---|---|---|---|
| L0 jtop | GPU 忙不忙？ | GPU 负载 46%、最忙的 CPU 核 78%、GPU 15.4 W | GPU 没吃满 → 看 CPU 侧 |
| L1 cProfile | CPU 时间花在哪个函数？ | postprocess 21.8%、_preprocess_python 15.0% | 两个热点都在 Python 侧 |
| L2 torch.profiler | 是哪些算子？有没有同步？ | 每步 2359 次 aten 调用；`aten::item` 14 次/step | 后处理是大量小算子 + 同步 |
| L3 nsys | GPU 时间线上发生了什么？ | GPU busy 34.2%；postprocess 397 kernel/step（98% < 5 μs）；infer GPU 3.158 ms | 后处理 Launch-bound；推理是 GPU 上最大的一块 |
| L4 ncu | 推理里哪个 kernel 慢、为什么？ | fp32 GEMM 47.7%，Occupancy 16.7%，Tail Effect（第 05 章 5.9） | fp16 / TensorRT |

**诊断决定修复顺序**：后处理（最大的一块，Launch-bound）→ 预处理（CPU-bound）→ 同步 → 推理精度 → 推理引擎。`no_pin`/`realloc` 在诊断中没有任何证据，按理不该优先处理；S5 仍然做了，用来演示"凭直觉的优化"。

## 8.2 逐项修复

[02_optimize.py](02_optimize.py)，日志 [outputs/02_optimize.log](outputs/02_optimize.log)，精度 [outputs/03_accuracy.log](outputs/03_accuracy.log)。

| 阶段 | 改动 | p50 ms | 本步 | 累计 | mJ/step | 对应章节 |
|---|---|---|---|---|---|---|
| S0 | 基线 | 10.321 | | 1.00x | 353 | |
| S1 | 后处理向量化 | 7.503 | 1.38x | 1.38x | 291 | 第 04 章 Launch-bound、第 07 章 7.2 |
| S2 | 预处理：只拷贝 uint8，归一化放到 GPU | 4.935 | 1.52x | 2.09x | 228 | 第 02 章、第 07 章 7.1 |
| S3 | 去掉循环中的 `.item()` | 4.641 | 1.06x | 2.22x | 222 | 第 03 章、第 07 章 7.2 |
| S4 | fp16 | 3.277 | 1.42x | 3.15x | 121 | 第 05 章 5.8 |
| S5 | pinned + 预分配 | 3.341 | 0.98x | 3.09x | 111 | 反例：第 04 章 4.8、第 07 章 7.4 |
| S6 | channels_last | 3.271 | 1.02x | 3.16x | 106 | 第 07 章 7.3 |
| S7 | torch.compile reduce-overhead | 1.819 | 1.80x | 5.67x | 69 | 第 03 章 |
| S8 | TensorRT fp16 + CUDA Graph（替代 S7） | 1.207 | 1.51x | 8.55x | 42 | 第 06 章 |
| S9 | resize 也放到 GPU | **0.747** | 1.62x | **13.82x** | **32** | 第 07 章 7.1 |

![stages](outputs/stages.png)

几个值得注意的点：
- **瓶颈一直在转移**：S0 时三个阶段差不多大；S2 之后推理成了大头（4.1 ms）；S8 之后预处理（0.77 ms，主要是 `cv2.resize`）又成了最大的一块，S9 把它也放到 GPU 上。每做完一步都要重新看各阶段的分布。
- **预期和实测对得上**：S1/S2/S3 的收益和 `common/selftest.py` 中逐个关闭开关的结果（1.37x / 1.29x / 1.19x）量级一致；顺序不同，单步的数值也会不同，因为每个开关的收益取决于其他部分有多快（第 07 章 7.2 A2）。
- **功耗上升，能耗下降**：S2/S3 时 GPU 更忙了，总功耗从 34 W 升到 48 W，但每步能耗从 353 降到 222 mJ。对电池供电的机器人来说，应该看能耗而不是功耗。
- **最终 GPU busy 只有 37.5%**（nsys `--cuda-graph-trace=node`，每步 84 个 kernel，基线 517 个）：剩下的时间大多是 CPU 侧的调度和阶段之间的串行等待。再往下优化，要靠整条流水线的 CUDA Graph 化，或者让相邻帧的阶段重叠（REPORT.md 第 6 节）。

## 8.3 测量中踩到的坑

1. **nsys 看不到 CUDA Graph 内部的 kernel**：第一次计算最终阶段的 GPU busy 时得到 6.1%，和"推理就要 0.35 ms GPU 时间"矛盾。原因是没有加 `--cuda-graph-trace=node`，CUDA Graph 内的 kernel 不出现在 KERNEL 表中。加上之后是 37.5%。（第 04 章常见坑 5）
2. **单次测量的波动**：S5 在第一次完整运行中是 0.94x，看起来像负优化；与 S4 交替重复 3 次后，两者的 p50 只差 0.02~0.03 ms，结论是"没有差别"。0.95~1.05 之间的单次结果不能作为结论。
3. **尾延迟的定位**：S6~S8 的 p99 偶尔比 p50 高 2 ms 以上。先假设是 OpenCV 线程池（把线程数设为 1：没有变化，排除），再逐阶段计时，定位到预处理中 147 KB 的 H2D 拷贝（这个大小会走 Copy Engine，第 04 章 4.8 C）。单独测试也复现了：147 KB 拷贝的 p99 是 236 μs，256 KB 是 46 μs。机制仍是推测，已写入报告。
4. **诊断阶段的观察者效应**：cProfile 让程序变慢约 40%，nsys 下每步 14.3 ms（不开时 10.3 ms），所以诊断数据只看相对比例，延迟一律用不开 profiler 时的测量。

## 8.4 报告

- [report_template.md](report_template.md)：背景与目标 → 环境 → 基线 → 诊断（每层：工具 → 证据 → 结论）→ 优化记录（假设、预期、实测、精度、副作用）→ 无效/负优化 → 结果与剩余瓶颈 → 风险 → 复现。
- [REPORT.md](REPORT.md)：本章的实测报告，按模板填写。

## 练习

见 [exercises.md](exercises.md)。

## 小结

- 诊断：jtop 说"CPU 侧"，cProfile 说"后处理和预处理"，torch.profiler 说"小算子和同步"，nsys 说"后处理 Launch-bound、推理占 GPU 大头"，ncu 说"fp32 GEMM 效率低"。每一层都缩小了范围，**没有一个工具能单独给出全部答案**。
- 修复：按诊断顺序 9 步，13.8 倍；其中 2 步（S5、S6）几乎没有收益，S5 本来就没有诊断依据。
- 记录：延迟之外，还要记录 p99、能耗、MFU 和精度；测量中的错误（GPU busy 6.1%）和推测（尾延迟的机制）也要写进报告。
