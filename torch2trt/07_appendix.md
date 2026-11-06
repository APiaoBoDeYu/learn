# 第 7 章：附录 —— 本教程没覆盖的路线、排查工具、常见坑

## 7.1 本教程没覆盖的路线，以及它们在地图上的位置

你的原始问题是"还有什么其他路线"。前六章测了 6 条，这里补上剩下的 —— 包括
**为什么它们不该和前六条放在同一张表里比较**。

### TensorRT-LLM：不是第七条路线，是另一个层级

最容易混淆的一个。TensorRT-LLM 和本教程六条路线**不在同一个抽象层级上**：

| | 本教程的 6 条路线 | TensorRT-LLM |
|---|---|---|
| 输入 | 任意 `nn.Module` | **预定义的模型架构**（Llama、GPT、Qwen…） |
| 做什么 | 把你的图编译成 engine | 提供针对该架构手工优化的实现 |
| 优化对象 | kernel 融合、tactic 选型 | **+ KV cache 管理、continuous batching、paged attention、张量并行** |
| 你的自定义架构 | 支持 | **需要按它的框架重写** |

关键区别：LLM 推理的瓶颈**不在 kernel 效率**，而在 KV cache 的显存管理和请求
调度。TensorRT-LLM / vLLM 优化的是后者，那是本教程六条路线完全不碰的维度。

所以：**如果你在做 LLM 服务，不要走本教程的路线**，直接用 vLLM 或 TensorRT-LLM。
前六章的结论（包括那些加速比）对 LLM 场景不适用。

反过来，如果你的模型是 CNN、检测、分割、小 transformer，那 TensorRT-LLM 帮不上忙。

### ONNX Runtime：一条平行路线

`torch → ONNX → ONNX Runtime`（可配 TensorRT execution provider）。
本教程没覆盖因为它严格说不是"转 TensorRT"。

值得知道的点：
- ORT 的 **TensorRT EP** 会自动把能转的子图交给 TRT，剩下的用 ORT 自己的 kernel。
  这在架构上**和 Torch-TensorRT 的部分回退非常像**，只是发生在 ONNX 层。
- 如果你已经在产 ONNX，ORT 是一个**比直接 TRT 构建低得多的门槛**：不用管 profile、
  不用写运行时、跨平台（CPU/GPU/其他加速器）。
- 代价是通常拿不到纯 TRT engine 的极限延迟。

对已有 ONNX 路线的人，这是一条很值得测一下的中间选项。

### torch.compile 的其他后端

本教程 R1 用的是默认的 Inductor。还有：
- `backend="torch_tensorrt"`：Torch-TensorRT 作为 torch.compile 后端。
  和 R4 编译的是同样的东西，但入口是 JIT 风格。**想最少改代码接入 TRT 就用这个。**
- `backend="onnxrt"`、`backend="tvm"` 等。

查看全部：`torch._dynamo.list_backends()`。

### 手写 CUDA / Triton kernel

当瓶颈是**一个特定算子**而不是整张图时，前六条路线都帮不上你 —— 它们优化的是
图结构。这时要下到 kernel 层：
- **Triton**：Python 写 block 级 kernel，比 CUDA C++ 简单得多。
- **CUDA extension**：极限性能。
- 两者都能接进 torch.compile 和 TensorRT（后者需要写 plugin，第 10 章有一个完整的
  C++ `IPluginV3` 例子和 Python `tensorrt.plugin` 的实测）。

这是姊妹教程 `/root/data/learn/triton` 的主题。

### 量化路线（已在第 8 章补上实测）

本章最初写的时候这是本教程最大的缺口。**第 8 章已经补上了完整实测** ——
包括一个推翻了"量化必然更快"的结果（INT8 要 batch ≥ 32 才反超 FP16）。
下面这些工具的定位仍然有效：

- `nvidia-modelopt`（TensorRT Model Optimizer）：PTQ / QAT / 稀疏化，
  能产出带 Q/DQ 节点的 ONNX 或直接喂 Torch-TensorRT。
- TensorRT 的 INT8 校准 + tensor core 支持是成熟工业方案。
- PyTorch 侧的 `torchao` / `quantization` 在追，但生态成熟度还有差距。

第 8 章的实测结论比这句话更细：**INT8 的收益取决于 batch size**，
小 batch 下反而更慢。具体数字见第 8 章。

## 7.2 排查工具速查

### 查 ONNX 导出有没有问题

```python
import onnx
m = onnx.load("model.onnx")
onnx.checker.check_model(m)
print(onnx.helper.printable_graph(m.graph))    # 人类可读的图
```

看图结构强烈推荐 **Netron**（`pip install netron; netron model.onnx`），
浏览器里可视化，排查"这个算子怎么变成这样了"最快。

### 查 TensorRT 引擎内部

```bash
# 逐层耗时 —— 排查"为什么慢"第一步
trtexec --loadEngine=model.plan --dumpProfile --separateProfileRun

# 看层融合结果（哪些层被合并了）
trtexec --onnx=model.onnx --fp16 --verbose 2>&1 | grep -i fusion

# 试动态 shape
trtexec --onnx=model.onnx --fp16 \
        --minShapes=in0:1x3x224x224 \
        --optShapes=in0:8x3x224x224 \
        --maxShapes=in0:16x3x224x224
```

`--dumpProfile` 是本教程最推荐的单个工具。它会告诉你时间花在哪一层，
于是"engine 是黑盒"这个问题至少在层粒度上被解决了。

### 查 torch.export 抓到了什么

```python
ep = torch.export.export(model, inputs)
print(ep)                  # 完整 ATen 图
print(ep.graph_signature)  # 输入/输出/参数的对应关系
```

第 5 章那段 `mean → gt → ne → item` 就是这么读出来的。

### 查 Torch-TensorRT 分区结果

本教程 `routes.py` 的做法：

```python
n_trt = sum(1 for n in gm.graph.nodes if "_run_on_acc" in str(n.target))
n_fallback = sum(1 for n in gm.graph.nodes if "_run_on_gpu" in str(n.target))
```

或者直接看整张图：`print(gm.graph)`。也可以开日志：

```bash
export TORCH_TENSORRT_LOG_LEVEL=DEBUG
```

**把子图计数当成标准检查项。** 第 1 章那个 `maxerr 0.0` 陷阱就是靠它发现的。

### 查 torch.compile 做了什么

```bash
TORCH_LOGS=graph_breaks python3 script.py    # 哪里 graph break 了
TORCH_LOGS=output_code   python3 script.py   # 生成的 Triton kernel 源码
TORCH_LOGS=recompiles    python3 script.py   # 为什么重编译
```

`graph_breaks` 对第 5 章那类问题特别有用 —— 它会直接告诉你在哪一行断的。

## 7.3 常见坑清单

### 坑：TensorRT 转换"成功"但其实什么都没转

**症状**：数值误差 0.0，性能和 eager 一样。
**原因**：`min_block_size` 默认 5，小模型整张图被跳过。
**查**：看那行 WARNING，或者数子图。见第 1 章。

### 坑：engine 只接受一个固定 shape

**原因**：ONNX 导出没写 `dynamic_axes`，或者 TRT 构建没加 optimization profile。
**注意**：两处都要配，只配一处不行。

### 坑：optimization profile 的 `opt` 选错，安静地慢

TensorRT 只针对 `opt` 那个 shape 做 tactic 选型。
**`opt` 要设成线上最常见的 shape，不是 min/max 的中间值。**

### 坑：int64 输入的 dtype 不匹配

TensorRT 对 INT64 支持有限，engine 常期待 INT32 而 torch 给 int64。
见第 3 章 `TrtRunner` 里的 cast，以及那个必须保持 cast 结果存活的细节
（`self._keep = tensor`）—— 忘了它会读到已释放的显存。

### 坑：`Double` 常量

`clamp(-0.99, 0.99)` 这种字面量 trace 成 FP64，TensorRT 不支持。
加 `truncate_double=True`（dynamo）或 `truncate_long_and_double=True`（ts）。

### 坑：把测试当验证

第 5 章的核心教训。**用示例输入分布测一遍，不能证明 engine 正确。**
必须用走不同分支、不同 shape 的输入测。

### 坑：一边构建一边测延迟

第 1 章的教训。TensorRT 构建吃满 GPU，并行会得到慢几十倍的假数字，
而且很容易被误读成"这个平台就是慢"。

### 坑：benchmark 的参考模型和被测模型权重不同

第 2 章的教训。`cos_sim ≈ 0` 且 eager 自己也过不了自检 —— 就是这个。
固定随机种子。

### 坑：ONNX 里有自定义算子，engine 部署后起不来

**症状**：`Cannot deserialize plugin since corresponding IPluginCreatorInterface not found`。
**原因**：engine 里只记了 plugin 的名字，代码在 `.so` 里，部署时没加载。
**注意**：它在 `createExecutionContext` 才报错，只做 `deserialize_cuda_engine` 的健康检查
查不出来。`.so` 还和 GPU 架构绑定，换卡要重编。见第 10 章。

### 坑：`dynamo=True` 导出的 ONNX，TRT parse 报 `Failed to import initializer`

**原因**：新导出器默认把权重写到 `.onnx.data` 外部文件，`parser.parse(bytes)` 找不到。
**解法**：用 `parser.parse_from_file(path)`，或导出时 `external_data=False`。

### 坑：假设"构建成功"就等于"能跑"

第 4 章的教训（`customop / ts_trt`）。构建通过、推理时 C++ 运行时崩。
benchmark 代码要把这两件事分开 try/except。

## 7.4 版本对齐速查

| 组合 | 怎么确定 |
|---|---|
| torch ↔ torch-tensorrt | 主次版本对齐。torch 2.9.x → torch-tensorrt 2.9.x |
| torch-tensorrt ↔ tensorrt | 看 wheel 元数据的 pin；小版本超出通常能用，**必须冒烟测试确认** |
| ONNX opset ↔ tensorrt | TRT 10.x 用 opset 17 稳妥。太低导不出新算子，太高 parser 不认 |
| torch ↔ triton | 随 torch 走，别单独升 |

安装 torch-tensorrt 的完整命令（含第 1 章那两个坑的解法）：

```bash
pip3 install --no-deps torch-tensorrt==2.9.0   # --no-deps 防止动 torch/tensorrt
pip3 install dllist                            # --no-deps 漏掉的隐式依赖
python3 -c "import torch_tensorrt as tt; print(tt._features._enabled_features_str())"
```

## 7.5 延伸阅读

- Torch-TensorRT 文档：https://pytorch.org/TensorRT/
- TensorRT 开发者指南：https://docs.nvidia.com/deeplearning/tensorrt/
- torch.export 教程：https://pytorch.org/docs/stable/export.html
- AOTInductor：https://pytorch.org/docs/stable/torch.compiler_aot_inductor.html
- TensorRT Model Optimizer（量化）：https://github.com/NVIDIA/TensorRT-Model-Optimizer
- 姊妹教程（Triton / torch.compile 内部机制）：`/root/data/learn/triton`

## 7.6 如果要继续做实验

本教程的基础设施已经支持但没跑的方向，按价值排序：

1. ~~INT8 量化对比~~ → **已完成，见第 8 章**（INT4 和 SmoothQuant 仍未测）。
2. ~~动态 shape 下的对比~~ → **已完成，见第 9 章**（多个动态维度叠加仍未测）。
3. **更大的模型**。规律 1 说明加速比随模型增大而提高，本教程最大只有 11.69M 参数，
   结论可能偏保守。换成 resnet50 / ViT-Base 值得一试。
4. **吞吐而非延迟**。多 stream、多 execution context、大 batch。
5. **ONNX Runtime + TensorRT EP** 作为第七条路线加进表里。

加一条新路线只需要在 `routes.py` 里写一个 `build(model, inputs, **kw) ->
(runnable, meta)` 函数并注册到 `ROUTES` 字典，其余（计时、校验、表格）自动接上。

---

← [第 6 章：综合对比与选型](06_benchmark_comparison.md) | [第 8 章：量化](08_quantization.md) →
