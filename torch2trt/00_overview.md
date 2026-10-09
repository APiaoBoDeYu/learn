# 第 0 章：全局地图 —— 从 PyTorch 到 GPU 推理，到底有几条路

> 本教程假设你已经在用 **torch → ONNX → TensorRT** 这条路线，并且想知道：
> 还有哪些路？它们各自的代价是什么？什么时候该换？
>
> 这一章先给出地图和判断框架，后面每一章用实测数据验证或推翻这里的说法。

## 0.1 先把问题问准确

"把 torch 模型转成 TensorRT engine 加速推理"这句话里其实藏了**两个独立的决策**，
把它们混在一起谈是选型出错的最常见原因：

**决策 A：用什么编译器 / 运行时？**
- TensorRT（NVIDIA 闭源，图级优化 + tactic 自动选型）
- Inductor（PyTorch 自带，生成 Triton kernel）
- 别的（ONNX Runtime、TVM……本教程不覆盖）

**决策 B：怎么把 PyTorch 的图交给它？**
- 经 ONNX（多一次 IR 转换，但格式通用）
- 经 torch.export（PyTorch 原生 IR，语义保真）
- 经 TorchScript（老方案）

你现在的路线是「A=TensorRT，B=经 ONNX」。它不是唯一组合，也不一定是最好的组合。
把两个决策拆开看，路线空间就清楚了：

|  | 经 ONNX | 经 torch.export | 经 TorchScript |
|---|---|---|---|
| **交给 TensorRT** | **R2** onnx_trt / **R3** trtexec | **R4** dynamo_trt | **R5** ts_trt |
| **交给 Inductor** | — | **R6** AOTInductor | — |
| **不编译** | — | — | **R1** torch.compile（JIT）/ **R0** eager |

本教程就是把这 7 个格子里的 6 条真实路线（R0–R6）在同一套模型、同一套计时代码下
全部跑一遍。

## 0.2 六条路线一览

| | 路线 | 一句话 | 产出物 | 运行时需要 Python 吗 |
|---|---|---|---|---|
| R0 | **eager** | 什么都不做 | 无 | 是 |
| R1 | **torch.compile** | Inductor JIT 生成 Triton kernel | 进程内缓存 | 是 |
| R2 | **onnx_trt** | `torch.onnx.export` → TRT Python builder | `.plan` | **否** |
| R3 | **trtexec** | 同一个 ONNX，用 CLI 构建 | `.plan` | **否** |
| R4 | **dynamo_trt** | `torch.export` → Torch-TensorRT | GraphModule / `.ep` | 是（除非导出 engine） |
| R5 | **ts_trt** | `torch.jit.trace` → Torch-TensorRT | TorchScript | **否**（C++ 可加载） |
| R6 | **AOTInductor** | `torch.export` → Inductor AOT 编译 | `.pt2` / `.so` | **否** |

最后一列常常是真正的决定因素。如果你的服务是 C++ 写的、或者要塞进一个不装 PyTorch
的容器，R1 和 R4 直接就出局了 —— 跟性能无关。

## 0.3 数据流对比图

你现在的路线（R2/R3），两次 IR 转换：

```
nn.Module
   │  torch.onnx.export   ← 坑集中在这一步：opset 覆盖、动态轴、trace 语义丢失
   ▼
ONNX 图 (protobuf 文件)
   │  TRT OnnxParser + Builder  ← 坑集中在这一步：不支持的算子、profile、精度
   ▼
TensorRT engine (.plan)  ← 与 GPU 架构 + TRT 版本强绑定
```

Torch-TensorRT 路线（R4），一次转换：

```
nn.Module
   │  torch.export        ← PyTorch 原生 IR，语义保真；控制流不再被静默抹平
   ▼
ExportedProgram (ATen 图)
   │  torch_tensorrt.dynamo.compile
   │     ├── 算子分区：能转的 → TRT engine
   │     └── 不能转的 → 留在 PyTorch  ← 关键差异：不是全有或全无
   ▼
GraphModule（内嵌一个或多个 TRT engine）
```

AOTInductor 路线（R6），完全不碰 TensorRT：

```
nn.Module
   │  torch.export
   ▼
ExportedProgram
   │  aoti_compile_and_package  → 生成 Triton kernel + C++ wrapper，编译成 .so
   ▼
.pt2 包（自包含，C++ 可直接加载）
```

**注意 R4 图里那个分区步骤**。这是 Torch-TensorRT 相对 ONNX 路线最实质的架构差异：
ONNX→TRT 默认是全有或全无（一个算子不支持，整条路断掉），而
Torch-TensorRT 会把图切开，能转的转、不能转的留在 PyTorch。第 5 章会实测这件事
——包括它的代价：每个 engine 边界都是一次同步 + 一次显存往返，切得太碎可能比 eager 还慢。

"全有或全无"只是**默认行为**，不是死路。ONNX 路线有两个对应的逃生口：导出期用
**自定义 ONNX 算子**（或把算子分解成标准算子），构建期给 TensorRT 一个 **plugin**。
它们要花工程成本，但能让产物保持完整的单个 engine。第 10 章会把这两种做法和
部分回退放在一起实测。

## 0.4 三个关键类比（帮你把新路线挂到已有认知上）

**类比 1：ONNX 之于 TensorRT，就像 C 之于汇编器。**
ONNX 是一个通用中间格式，好处是任何前端都能产出它、任何后端都能消费它；代价是它有
自己的算子集，PyTorch 的语义要先翻译成 ONNX 算子，翻译不过去的就丢了(或者你自定义onnx算子)。
`torch.export` 走的是另一条思路：不翻译，直接把 PyTorch 自己的 ATen 图交给后端。
**少一次翻译，就少一类 bug。**

**类比 2：`min_block_size` 之于 Torch-TensorRT，就像内联阈值之于编译器。**
它决定"多大的子图才值得单独建一个 engine"。太大 → 什么都不转（第 1 章那个
`maxerr 0.0` 陷阱）；太小 → 切出一堆小 engine，边界开销吃掉收益。

**类比 3：torch.compile 之于 TensorRT，就像 JIT 之于 AOT。**
这个类比直接决定了它们各自的适用场景，而且和性能无关：
- torch.compile 首次调用才编译，shape 变了就重编译，但**永远不会因为"转不了"而失败**
  （最差就是 graph break 退回 eager）。
- TensorRT 离线构建，engine 与 shape/架构/版本绑定，但**运行时零编译开销、可脱离 Python**。

## 0.5 一张"什么情况选什么"的预判表

这是**基于文档和经验的先验判断**，不是实测结论。教程后面每一章都会去验证它，
第 6 章会给出修正版 —— 有几条判断会被数据推翻，那正是做实验的意义。

| 你的情况 | 预判推荐 | 理由 |
|---|---|---|
| 纯 CNN，静态 shape，要极限延迟 | R2/R3（ONNX→TRT） | TRT 的 conv 融合和 tactic 库在这类模型上最成熟 |
| 模型里有 TRT 不支持的算子 | R4（dynamo_trt） | 自动部分回退，不用写 plugin（R2/R3 + 自定义算子/plugin 见第 10 章） |
| 有数据相关控制流 | R1（torch.compile） | graph break 保证正确性 |
| 要部署到 C++ 服务、不能带 Python | R2/R3 或 R6 | 产出自包含的 `.plan` / `.so` |
| 训练也想加速 | R1 | TensorRT 只做推理 |
| 想最快看到收益、不想改架构 | R1 | 一行 `torch.compile(model)` |
| 目标是 LLM 推理 | 都不是 —— 见第 7 章 | vLLM / TensorRT-LLM 是另一个层级 |

## 0.6 本教程怎么验证这些说法

四个模型，每个专门打一个痛点：

| 模型 | 考验什么 | 为什么需要它 |
|---|---|---|
| `resnet18` | 对照组：所有路线都该能跑 | 没有它，任何加速比都没有基准 |
| `bert_tiny` | matmul/layernorm/softmax + 动态 shape | CNN 上的结论不能直接套到 transformer |
| `ctrlflow` | 数据相关 `if` | 分辨"静默出错"和"诚实报错"的唯一方法 |
| `customop` | 无 converter 的算子（`erfinv`） | 量化"不支持算子"这件事的真实代价 |

三条方法论纪律，贯穿全部实验：

1. **正确性一律和 fp32 eager 比**，不和 fp16 eager 比 —— 否则一个 fp16 路线会因为
   "和另一个 fp16 路线误差相同"而白拿分。
2. **不只看 `allclose`**。FP16 下单个离群元素就能让它 fail，而模型预测毫无变化。
   所以同时报 cosine 相似度和 argmax 一致率 —— 后者才是决定能不能上线的指标。
3. **构建时间算进成本**，包含 ONNX 导出。那是你在 CI 里真实等待的时间。

---

[第 1 章：环境搭建](01_setup.md) →
