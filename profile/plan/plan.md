# Plan：技术方案（How）

> 依据 [spec.md](spec.md)。本文给出目录结构、共享组件、每章内容设计、执行与验证方法、风险。

## 1. 目录结构

```
profile/
├── README.md                    # 大纲（完成后各章加链接）
├── requirements.txt             # 额外依赖
├── doc/                         # 已有 xmind 笔记
├── plan/                        # spec / plan / tasks / 进度日志
├── common/                      # 跨章节共享代码
│   ├── bench_utils.py           # 计时 + 统计 + 环境快照
│   ├── models.py                # 示例模型
│   ├── pipeline.py              # "慢流水线"（可切换各种低效开关）
│   └── env_info.py              # 打印 / 记录运行环境
├── 00_theory/
├── 01_system_monitor/
├── 02_python_profilers/
├── 03_torch_profiler/
├── 04_nsight_systems/
├── 05_nsight_compute/
├── 06_tensorrt/
├── 07_optimization/
├── 08_case_study/
└── appendix/
```

每章目录统一为：
```
NN_xxx/
├── README.md        # 正文
├── NN_xx_demo.py    # 可运行示例（编号排序）
├── run.sh           # 可选：一键跑完本章命令行工具（nsys/ncu/trtexec）
├── exercises.md     # 练习 + 参考思路
└── outputs/         # 运行产物（.gitignore 忽略）
```

## 2. 共享组件设计（common/）

### 2.1 `bench_utils.py`
- `benchmark(fn, warmup=10, iters=50, sync=True) -> Stats`：用 `torch.cuda.Event` 计时，返回 mean/p50/p90/p99/std/吞吐
- `cpu_timer` 上下文管理器：用 `time.perf_counter` 计时，并演示"不同步计时"会测错
- `compare(results: dict[str, Stats])`：打印 baseline 与优化版本的对比表（含加速比）
- `env_snapshot()`：torch/CUDA 版本、GPU 名称、nvpmodel 模式、GPU 频率、温度（尽力而为，失败就跳过）

### 2.2 `models.py`
- `TinyCNN`：类 ResNet 小网络，作为视觉 backbone 和 TensorRT 导出对象
- `TinyTransformer`：几层 Encoder（用来演示 Attention、prefill/decode、torch.compile）
- `ManySmallOps`：大量小算子串联，专门制造 Launch-bound
- `ToyVLA`：CNN 视觉编码 + Transformer + 动作头，近似 `../vla` 的结构，供第 08 章使用

### 2.3 `pipeline.py`：可切换低效开关的"慢流水线"
流程：`读取/生成图像 → 预处理 → 推理 → 后处理`。开关如下：

| 开关 | 低效行为 | 对应瓶颈 | 定位工具 | 修复章节 |
|---|---|---|---|---|
| `py_preprocess` | Python 逐像素循环归一化 | CPU-bound | VizTracer / py-spy | 02、07 |
| `sync_in_loop` | 每步 `.item()` / `print(tensor)` | 同步 | torch.profiler / nsys | 03、04 |
| `no_pin` | 不用 pinned memory，同步拷贝 | 拷贝 | nsys | 04 |
| `fp32` | 不用半精度 | Compute/Memory | ncu | 05、07 |
| `eager_small_ops` | 后处理拆成大量小算子 | Launch-bound | nsys | 04、07 |
| `realloc` | 每次迭代新建大张量 | 分配/碎片 | memory snapshot | 03 |

第 08 章从"开关全开"开始，逐个定位、逐个关闭，记录每一步的收益。

## 3. 各章内容设计要点

| 章 | 关键 demo | 要验证的现象 |
|---|---|---|
| 00 | `01_wrong_vs_right_timing.py`、`02_roofline.py`（实测 GEMM/elementwise 的 FLOPs/s 和带宽，画 Roofline）、`03_mfu.py` | 不同步计时会测错；elementwise 落在带宽区，大 GEMM 接近算力屋顶 |
| 01 | `01_jtop_logger.py`（jtop API 采样 → CSV → 图）、`02_dmon_logger.py`、`03_load_patterns.py`（CPU 型负载 vs GPU 型负载） | GPU% 和 EMC% 在两类负载下的差异；nvpmodel / 频率对延迟的影响 |
| 02 | cProfile、line_profiler、py-spy、VizTracer 分析同一个 `py_preprocess` 流水线 | 四种工具的视角和开销对比；VizTracer 看到的 GPU 调用耗时"偏短"（异步） |
| 03 | 基础表格、trace + schedule、record_function、memory snapshot、同步点检测、torch.compile graph break | Top-N 算子；`.item()` 在 trace 中表现为 `cudaStreamSynchronize`；碎片可视化 |
| 04 | NVTX 标注流水线、`nsys stats`、多 stream overlap、CUDA Graph 前后对比、GPU metrics 尝试 | GPU bubble 的量化；overlap 前后的时间线；Graph 消除 launch gap |
| 05 | 用 Triton / torch 自定义 kernel 构造 4 类瓶颈 + bank conflict（`torch.utils.cpp_extension.load_inline` 写 CUDA） | 每类瓶颈在 ncu 中的特征指标与修复后的变化 |
| 06 | 导出 TinyCNN 到 ONNX → trtexec FP32/FP16/INT8 → dumpProfile → nsys → ncu | 三步法完整走通；reformat 层的识别；layer 与 kernel 的对应 |
| 07 | 每个优化手段一个最小前后对比 | 量化加速比；说明哪些手段对哪类瓶颈无效（例如 CUDA Graph 对 Memory-bound 无效） |
| 08 | `ToyVLA` + 慢流水线全流程 + `report_template.md` + 实测报告 | 串起所有工具，输出完整报告 |
| 附录 | 速查表、工具对比、常见坑 | 汇总前面各章实测中踩到的坑 |

## 4. 执行方法（面向规格的迭代）

每个任务按以下循环执行：

```
读 spec 对应条目 → 写 demo → 本机运行 → 收集真实输出 → 写 README → 对照 DoD 自检 → 更新 tasks.md（状态 + 发现）
```

- **顺序**：T00（环境）→ T01（common）→ 各章按 00→08 顺序 → 附录 → 收尾。common 是所有章的依赖，必须先完成；第 08 章依赖前面所有章。
- **GPU 独占**：profiling 和 benchmark 需要独占 GPU，因此各章**串行执行**，不并行跑 demo（否则数据互相干扰，ncu 也会串行化 kernel）。
- **进度记录**：每完成一个任务，在 `tasks.md` 中勾选并在"实测记录"写下关键发现。出现偏离 spec 的情况（例如工具不支持），先更新 spec 的 Open Questions，再继续。
- **检查点**：完成 T00+T01、第 03 章、第 05 章、第 08 章时，向用户汇报一次进度，用户确认后再继续。

## 5. 验证方法

- 每个 demo：`/opt/venv/bin/python NN_xx.py` exit 0，耗时满足 NFR-2。
- 文档中的输出用 `tee outputs/xxx.log` 保存，再从日志中摘录，保证真实。
- 每章完成后通读一遍 README，检查命令是否可复制粘贴运行、路径是否正确。

## 6. 风险与应对

| 风险 | 影响 | 应对 |
|---|---|---|
| 无法联网 pip 安装 | 02/03 部分工具没法实操 | 改为文档说明 + 用已有工具替代（cProfile、torch.profiler） |
| Thor 上 nsys GPU metrics / ncu 部分 section 不支持 | 04/05 内容缩水 | 如实记录；用 ncu 的 SOL/Occupancy 等可用部分替代 |
| 容器内无法锁频或切换 nvpmodel | 结果波动 | 多次测量取中位数，并注明未锁频 |
| ncu 采集慢 | 超时 | `-k` + `--launch-count 1` + 指定 `--section` |
| TensorRT Python 包缺失 | 06 无法用 Python API | 统一用 trtexec 命令行（已确认存在） |
| profile 产物占用空间 | 磁盘 | 产物放 outputs/，章节完成后清理大文件 |
