# Spec：模型性能分析与调优教程

> 阶段：规格定义（What & Why）。不涉及具体实现细节，实现方案见 [plan.md](plan.md)，任务拆解见 [tasks.md](tasks.md)。
> 依据：[../README.md](../README.md)（教程大纲）+ [../doc/](../doc/)（已有 xmind 笔记）

## 1. 背景与目标

- **背景**：做模型调优和推理优化时，要先确定性能瓶颈。工具链有很多（VizTracer、torch.profiler、Nsight Systems、Nsight Compute、jtop、TensorRT Profiler……），但缺少一份把它们串起来的系统教程。
- **目标**：按大纲生成一套**可以跑起来**的中文教程。学完之后读者能够：
  1. 说清楚 GPU 利用率各项指标的含义，以及四大瓶颈类型（加上 CPU 瓶颈和同步瓶颈）怎么判定；
  2. 熟练使用 L0–L4 各层工具，独立完成一次自顶向下的瓶颈定位；
  3. 针对诊断出的瓶颈选对优化手段，并量化优化收益。

## 2. 读者

- 有 PyTorch / CUDA 基础，熟悉 Jetson，已有 CUDA 和 TensorRT 的笔记（见 doc/）。
- 主要缺口：Python 层和框架层 profiling 的实操经验，以及工具之间如何联动。

## 3. 运行环境（约束）

| 项 | 值 |
|---|---|
| 硬件 | NVIDIA Jetson Thor（aarch64，CPU/GPU 统一内存） |
| 系统 | L4T R38.4 / JetPack 7 |
| Python | `/opt/venv/bin/python`（torch 2.9.1，CUDA 可用） |
| 工具 | nsys 2025.3、ncu 2025.3、jtop、nvidia-smi、trtexec |
| 用户 | root（ncu 计数器权限理论上不受限，需验证） |

## 4. 范围

### 4.1 包含（In Scope）

与大纲一一对应，共 9 章 + 附录：

| 章 | 主题 | 核心产出 |
|---|---|---|
| 00 | 基础理论 | 利用率指标、Roofline、瓶颈分类、正确计时；`bench_utils.py` |
| 01 | 系统级监控 | jtop / tegrastats / nvidia-smi dmon / 指标采集脚本 |
| 02 | Python 级 | cProfile / line_profiler / py-spy / VizTracer |
| 03 | torch.profiler | 算子表、trace、schedule、显存分析、HTA、torch.compile 诊断 |
| 04 | Nsight Systems | 采集、NVTX、stats/analyze、stream overlap、CUDA Graph |
| 05 | Nsight Compute | SOL/Roofline/Occupancy/Memory/Stall；四类瓶颈 kernel |
| 06 | TensorRT | trtexec、TRT Profiler、三步法 |
| 07 | 优化手段 | "瓶颈 → 手段" 对照表中每种手段的前后对比 |
| 08 | 综合实战 | 用同一条"慢流水线"走完全流程并逐步优化，产出报告 |
| 附录 | 速查与避坑 | 命令速查、工具对比、可视化流程、常见坑、参考资料 |

### 4.2 不包含（Out of Scope）

- 训练侧分布式（NCCL、多卡通信）只点到为止，不展开实践（单机 Jetson 无条件）。
- 不系统重讲 CUDA 编程基础，引用 doc/CUDA.xmind 即可。
- 不保证 x86 dGPU 环境下的命令输出，只在文中注明差异。

## 5. 功能需求（每章必须满足）

- **FR-1 文档结构**：每章 `README.md` 包含：学习目标 → 原理 → 工具用法 → 示例演练（命令 + **真实输出节选** + 解读）→ 常见坑 → 练习题 → 小结 / 📎 笔记引用。
- **FR-2 可运行代码**：每章至少 2 个 demo 脚本，用 `/opt/venv/bin/python` 在本机能直接运行，并在文档中给出运行命令。
- **FR-3 真实结果**：文中的输出、数字和截图描述来自本机实测，不编造。工具在 Thor 上不支持或受限时，如实写明并给出替代方案。
- **FR-4 前后对比**：涉及优化的 demo 要有 baseline 与 optimized 两版，用统一的 benchmark 工具量化收益。
- **FR-5 连贯案例**：各章尽量复用同一套示例模型和流水线（见 plan 中的 `common/`），让读者看到同一个问题在不同工具下的表现。
- **FR-6 产物管理**：profile 产物（`.nsys-rep`、`.ncu-rep`、`.json` trace、`.html` 等）统一输出到各章的 `outputs/` 目录，不放进正文；体积大的产物不长期保留。
- **FR-7 练习题**：每章 3–5 道练习，至少 1 道动手题，并附参考答案思路（折叠或单独文件）。

## 6. 非功能需求

- **NFR-1 语言**：正文简体中文；命令、代码、报错、工具字段名保持原文。
- **NFR-2 时长**：单个 demo 运行时间 ≤ 2 分钟（ncu 全量采集可放宽，但必须用过滤参数控制）。
- **NFR-3 可复现**：benchmark 前固定频率（或注明未锁频），记录 nvpmodel 模式、温度等上下文。
- **NFR-4 自包含**：demo 只依赖 torch 以及少量可 pip 安装的包（viztracer、py-spy、line_profiler、nvtx、HTA 等），依赖统一写在 `requirements.txt`。
- **NFR-5 篇幅**：每章 README 控制在易读范围（约 300–800 行）；细节放到子文档或代码注释里。

## 7. 验收标准（Definition of Done）

一章完成需满足：
1. [ ] README 符合 FR-1 结构
2. [ ] 所有 demo 在本机运行成功（exit 0），关键输出已贴入文档
3. [ ] 优化类 demo 有量化对比数据（FR-4）
4. [ ] 有练习题和参考思路（FR-7）
5. [ ] 不支持/受限的功能已说明
6. [ ] tasks.md 中对应任务勾选，并记录实测中的偏差和发现

整体完成：9 章 + 附录全部达到 DoD；顶层 README 的大纲链接到各章；第 08 章报告能串起前面所有工具。

## 8. 待确认问题（Open Questions）

| # | 问题 | 处理方式 |
|---|---|---|
| Q1 | pip 能否联网安装 viztracer / py-spy / HTA 等？ | ✅ 可以联网（T00）；新容器重装，jtop 客户端需从 GitHub 提交安装（env_report） |
| Q2 | Thor 上 `nsys --gpu-metrics-devices` 是否可用？ | 旧容器 ❌（Insufficient privilege）；**新 privileged 容器 ✅**（列出 `Blackwell GB10B \| NVIDIA Thor`），2026-11-06 |
| Q3 | Thor 上 ncu 计数器权限、`--clock-control` 行为 | 旧容器 ❌（ERR_NVGPUCTRPERM）；**新 privileged 容器 ✅**。`--clock-control base` 在 Thor 上不生效（报 `without fixed GPU frequencies`），与 `none` 结果一致，频率由宿主机 jetson_clocks 锁在 1.57 GHz；`dram__*` 指标为 n/a（详见第 05 章） |
| Q4 | tegrastats 是否存在（当前 PATH 中没有） | ❌ 容器内没有（新旧容器都一样），只在宿主机上采集了样例；容器内用 jtop API |
| Q5 | 第 08 章用 toy 流水线还是 `../vla` 的真实模型？ | ✅ 用 toy 流水线（ToyVLA + SlowPipeline）；真实 VLA 留作第 08 章练习 4 |
| Q6 | 是否有 sudo/锁频权限（`jetson_clocks`、`nvpmodel`）| 容器内没有这些命令（新容器也一样），宿主机已是 MAXN + jetson_clocks；修改功耗模式需征得用户同意 |
