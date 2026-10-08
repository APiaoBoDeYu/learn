# torch → TensorRT：六条转换路线的系统对比

面向**已经在用 torch → ONNX → TensorRT 的读者**：这个教程回答"还有哪些路线、
它们各自的优缺点是什么、什么时候该换"。

所有代码在真实环境跑通验证，文档里每个延迟数字都是本机实测，不是估算。
包括那些**推翻了我事先判断**的结果 —— 教程里会明确标出哪些先验被数据否掉了。

## 最重要的六个结论（如果只看一段）

1. **静态 shape 下，迁移到 Torch-TensorRT 换不来性能。** resnet18 / bert_tiny 上
   dynamo_trt 和 ONNX 路线差异在 5% 以内 —— 因为两者最终都是同一个 TensorRT
   builder 在优化。迁移的理由是能力差异（部分回退、控制流严格性）。
   **但动态 shape 下这条不成立，见结论 5。**
2. **ONNX 和 TorchScript 路线在数据相关控制流上会静默产出错误的模型。**
   实测：被 trace 的分支 cosine = 1.00000，另一个分支 cosine = **0.386**，
   构建过程零警告，而且延迟表上排第一。基于 `torch.export` 的路线会直接拒绝。
3. **AOTInductor 被严重低估。** 构建最快（比 TRT 路线快约 2 倍）、2.3–2.5x 加速、
   产物脱离 Python、**零额外依赖**。在含不支持算子的模型上它是全表最快的路线 ——
   即使 ONNX 路线补上自定义算子 + TensorRT plugin 之后也是。（第 5、10 章）
4. **INT8 不是无条件更快。** batch ≤ 8 时 INT8 **比 FP16 慢**（0.89–0.97x），
   要到 batch ≥ 32 才反超（1.14–1.18x）；准确率只掉 0.13pp。
   FP8 在这块硬件上全程是净损失（0.44–0.80x）。（第 8 章）
5. **动态 shape 改变了结论 1。** 静态 shape 下各 TRT 路线差 <5%，但动态 shape 下
   dynamo_trt 比 ONNX 路线快 **20–29%**。而且动态 engine 比专用静态 engine
   慢 **44–84%** —— shape 可枚举时应该建多个静态 engine。（第 9 章）
6. **ONNX 路线不是"全有或全无"。** 不支持的算子有两个逃生口：导出期的自定义 ONNX 算子、
   构建期的 TensorRT plugin。实测最好的方案反而**不写 plugin**：在导出期把算子分解成
   标准 ONNX 算子，TensorRT 能把它和邻居融成一个 kernel，比手写 C++ plugin 快 11%。
   Python plugin（QDP）在 TRT 10.14 上每次推理都重载 PTX，慢 40 倍。（第 10 章）

## 环境

```
硬件     : NVIDIA Thor (Jetson, aarch64), compute capability (11, 0) / sm_110a
CUDA     : 13.0
PyTorch  : 2.9.1
TensorRT : 10.14.1.48
torch-tensorrt : 2.9.0
torchvision    : 0.24.1
nvidia-modelopt: 0.47.0   (第 8 章量化用)
```

换机器请重跑 —— 绝对数字一定变，趋势大概率保持。第 6 章有适用边界的诚实说明。

## 六条路线

| | 路线 | 一句话 | 产出物 | 运行时要 Python 吗 |
|---|---|---|---|---|
| R0 | eager | 基准 | — | 是 |
| R1 | torch.compile | Inductor JIT → Triton kernel | 进程内缓存 | 是 |
| R2 | onnx_trt | `torch.onnx.export` → TRT Python builder | `.plan` | **否** |
| R3 | trtexec | 同一个 ONNX，CLI 构建 | `.plan` | **否** |
| R4 | dynamo_trt | `torch.export` → Torch-TensorRT | GraphModule | 是 |
| R5 | ts_trt | `torch.jit.trace` → Torch-TensorRT | TorchScript | **否** |
| R6 | aoti | `torch.export` → AOTInductor | `.pt2` / `.so` | **否** |

## 学习路线图

| 章节 | 内容 | 建议用时 |
|---|---|---|
| [00 全局地图](00_overview.md) | 把"转 TensorRT"拆成两个独立决策，六条路线的坐标，三个类比，预判表 | 25 分钟 |
| [01 环境搭建](01_setup.md) | 各路线依赖矩阵；装 torch-tensorrt 的四个坑；**"误差 0.0 其实什么都没转"陷阱** | 25 分钟 |
| [02 基准与模型设计](02_baseline_and_models.md) | benchmark 的四种骗法（含我真踩的）；为什么 `allclose` 不够；四个模型各证明什么 | 30 分钟 |
| [03 ONNX → TensorRT](03_onnx_trt_route.md) | 你现在的路线：两步两类坑；opset/profile/dtype；**你得自己写的那 40 行运行时**；R2 vs R3 实测 | 40 分钟 |
| [04 Torch-TensorRT](04_torch_tensorrt_route.md) | 六个入口的对照表；`min_block_size` 陷阱；**部分回退的能力与代价**；为什么别用 TorchScript 前端 | 45 分钟 |
| [05 控制流与回退](05_control_flow_and_fallback.md) | **全教程最重要一章**：静默出错 vs 诚实拒绝 vs graph break，三种行为实测 | 40 分钟 |
| [06 综合对比与选型](06_benchmark_comparison.md) | 4 模型 × 7 路线完整数据；**被推翻的四个先验判断**；决策流程；适用边界 | 40 分钟 |
| [07 附录](07_appendix.md) | TensorRT-LLM / ONNX Runtime 的定位；排查工具；坑清单；版本对齐 | 需要时查阅 |
| [08 量化](08_quantization.md) | INT8 / FP8 三种机制；**真实 top-1 精度损失**；为什么 INT8 要大 batch 才有收益；量化工具自己踩了第 5 章的坑 | 45 分钟 |
| [09 动态 shape](09_dynamic_shapes.md) | 三种声明方式对比；**动态 engine 比静态慢 44–84%**；动态下 dynamo_trt 反超 ONNX；一个污染全局的真 bug | 40 分钟 |
| [10 自定义算子与 plugin](10_custom_op_and_plugin.md) | ONNX 的两个逃生口：自定义算子 / 分解 + TRT plugin（C++ / Python）；**plugin 是融合屏障**；同一个 `.so` 也能救 dynamo_trt | 45 分钟 |

建议按顺序读。第 2 章的方法论是后面所有数据可信的前提；第 5、6 章的结论依赖前面所有章节。

## 快速开始

```bash
cd /root/data/learn/torch2trt/code

python3 common.py          # 环境自检
python3 models.py          # 确认四个模型都能前向

python3 run_bench.py --models resnet18          # 单模型全路线（约 2 分钟）
python3 exp_control_flow.py                     # 第 5 章核心实验
python3 exp_unsupported_op.py                   # 不支持算子实验
bash plugins/build.sh && python3 exp_custom_plugin.py   # 第 10 章：自定义算子 + plugin
python3 run_bench.py                            # 全矩阵（约 15 分钟）
python3 run_bench.py --table                    # 从已有结果重渲染表格
```

## 四个实验模型，每个负责证明一件事

| 模型 | 考验什么 | 为什么需要它 |
|---|---|---|
| `resnet18` (11.69M) | 对照组，TRT 最佳战场 | 给出加速比上界；出问题时排除环境因素 |
| `bert_tiny` (4.77M) | transformer；matmul/layernorm/softmax | CNN 上的结论不能套到 transformer |
| `ctrlflow` (0.53M) | 数据相关 `if` | **唯一能分辨"静默出错"和"诚实报错"的方法** |
| `customop` (0.27M) | 无 converter 的算子（`erfinv`） | 量化"不支持算子"的真实代价 |

## 目录结构

```
/root/data/learn/torch2trt/
├── README.md                          本文件
├── 00_overview.md ~ 10_custom_op_and_plugin.md   十一个章节
├── context.txt                        最初的问题（教程的起点）
├── code/
│   ├── common.py                  计时 / 正确性判据 / 结果落盘
│   ├── models.py                  四个实验模型
│   ├── routes.py                  七条路线，统一 build() 接口
│   ├── run_bench.py               benchmark 驱动 + 表格渲染
│   ├── exp_control_flow.py        第 5 章：控制流实验
│   ├── exp_unsupported_op.py      第 5 章：不支持算子实验
│   ├── data.py                    第 8 章：真实 ImageNet 数据 + top-1 评估
│   ├── quant_routes.py            第 8 章：INT8/FP8 三种量化机制
│   ├── exp_quantization.py        第 8 章：量化主实验
│   ├── exp_quant_batch.py         第 8 章：batch 扫描追加实验
│   ├── exp_dynamic_shape.py       第 9 章：动态 shape 实验
│   ├── exp_custom_plugin.py       第 10 章：ONNX 自定义算子 + TRT plugin
│   └── plugins/                   第 10 章：C++ IPluginV3 源码 + build.sh
├── data/imagenette2-160/          第 8 章用的真实图片（约 99 MB，需下载）
└── artifacts/                     ONNX / engine / .pt2 / results.jsonl
```

## 加一条新路线

`routes.py` 里写一个函数并注册，其余（计时、校验、表格）自动接上：

```python
def build_my_route(model, inputs, **kw):
    ...
    return runnable, {"build_s": secs}

ROUTES["my_route"] = build_my_route
```

路线处理不了某个模型时 `raise RouteUnsupported(...)` —— 那会被记成表格里的一个
`X`，而不是让整个 benchmark 崩掉。"这条路线拒绝了这个模型"本身就是实验结果。

## 姊妹教程

`/root/data/learn/triton` —— Triton 和 torch.compile 的内部机制。
当瓶颈是**单个算子**而不是整张图时，那边才是你要去的地方。
