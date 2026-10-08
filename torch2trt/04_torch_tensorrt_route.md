# 第 4 章：R4 / R5 —— Torch-TensorRT（dynamo 与 TorchScript 前端）

上一章那条路线有两次 IR 转换。本章这条只有一次：`torch.export` 直接把 PyTorch 的
ATen 图交给 TensorRT，中间不经过 ONNX。

本章要说明的是：**这不只是"少一步"这种量的差别，它带来了一个 ONNX 路线在架构上
做不到的能力 —— 部分回退。**

## 4.1 Torch-TensorRT 有几个入口

这是你在上一轮对话里问到的问题，先用一张表定位清楚：

| 入口 | 输入 | 输出 | 什么时候用 |
|---|---|---|---|
| `torch_tensorrt.compile(...)` | `nn.Module` / `ScriptModule` | 编译后的**可调用模块** | 通用入口，按 `ir=` 参数分派到下面几条 |
| `torch_tensorrt.dynamo.compile(...)` | `ExportedProgram` | `GraphModule` | **本教程 R4**，最显式可控 |
| `torch.compile(m, backend="torch_tensorrt")` | `nn.Module` | 可调用模型 | 已经在用 torch.compile，想最少改代码 |
| `torch_tensorrt.ts.compile(...)` | `ScriptModule` | TorchScript 模块 | **本教程 R5**，老项目兼容 |
| `convert_method_to_trt_engine(...)` | 模块方法 + 示例输入 | **序列化 engine bytes** | 你要的就是一个 `.plan` 文件 |
| `torchtrtc` CLI | TorchScript 模型文件 | TorchScript 或 PLAN | 命令行流程 |

关键区分：**`compile` 系列返回"能跑的 PyTorch 对象"，`convert_method_to_trt_engine`
返回"engine 字节"。** 如果你的目标是拿一个能被 C++ 服务加载的文件，用后者（或者
`torchtrtc`），别指望 `compile` 给你。

本教程 R4 用 `dynamo.compile`，因为它把每一步都摆在明面上：你自己调
`torch.export.export()`，能先检查 `ExportedProgram`，再交给 TRT。
`torch.compile(backend=...)` 更省事但把导出藏起来了，出问题更难查。

## 4.2 R4 的实现

```python
def build_dynamo_trt(model, inputs, fp16=True, min_block_size=1,
                     dynamic_shapes=None, **kw):
    import torch_tensorrt as tt
    precisions = {torch.half, torch.float} if fp16 else {torch.float}

    def do_build():
        ep = torch.export.export(model, inputs, dynamic_shapes=dynamic_shapes)
        return tt.dynamo.compile(
            ep, inputs=list(inputs),
            enabled_precisions=precisions,
            min_block_size=min_block_size,
            truncate_double=True,
            offload_module_to_cpu=True,
        )

    gm, build_s = _timed(do_build)
    n_trt = sum(1 for n in gm.graph.nodes if "_run_on_acc" in str(n.target))
    n_fallback = sum(1 for n in gm.graph.nodes if "_run_on_gpu" in str(n.target))
    ...
```

### 参数逐个说

**`min_block_size`（最重要）**：一个子图至少要有多少个算子才值得单独建 engine。
第 1 章那个 `maxerr 0.0` 陷阱就是它的默认值 5 导致的 —— 4 个算子的模型整张图被跳过，
一个 engine 都没建，误差当然是 0。

这个参数是一个真实的权衡，可以直接类比编译器的内联阈值：
- 太大 → 该转的没转，白白留在 PyTorch。
- 太小 → 切出一堆小 engine，每个边界一次同步 + 一次显存往返，开销吃掉收益。

本教程设 `min_block_size=1` 是为了让"转了多少"这件事可测，**不是生产推荐值**。
生产上应该从默认值开始，配合下面那两个计数去调。

**那两个计数（`trt_subgraphs` / `fallback_subgraphs`）**：这是本教程加的，不是
API 提供的。`_run_on_acc_*` 是 TRT engine 子图，`_run_on_gpu_*` 是留在 PyTorch 的
子图。**这两个数字应该成为你用这条路线时的标准检查项** —— 它把"转换到底生效了吗"
从一个假设变成一个可读的数字。

**`truncate_double=True`**：把 FP64 常量截成 FP32。TensorRT 不支持 FP64，不加这个
碰到 double 常量就报错。注意 `bert_tiny` 里 `clamp(-0.99, 0.99)` 这种字面量在
trace 时就是 double（上一章 ONNX 路线的图里能看到
`%22 : Double(device=cpu) = prim::Constant[value={-0.99}]`），所以这个参数比你想的
更常需要。R5 路线里同一个参数叫 `truncate_long_and_double`。

**`offload_module_to_cpu=True`**：第 1 章讲过，Jetson 统一内存下让 builder 在构建期
把权重暂存 host 侧，避开显存告警。

## 4.3 实测：R4 vs ONNX 路线

本机实测（FP16），中位延迟 ms：

| 模型 | eager | onnx_trt | trtexec | **dynamo_trt** |
|---|---|---|---|---|
| resnet18 (b=8) | 2.879 | 0.852 | 0.826 | **0.807** |
| bert_tiny (b=8,s=128) | 0.867 | 0.413 | 0.287 | **0.296** |
| customop | 0.068 | **构建失败** | **构建失败** | **0.202** |

构建耗时（s）：

| 模型 | onnx_trt | trtexec | dynamo_trt |
|---|---|---|---|
| resnet18 | 17.8 | 21.6 | 20.6 |
| bert_tiny | 13.1 | 18.5 | 15.3 |

**第一个结论，可能和你的预期相反：在能跑的模型上，R4 和 ONNX 路线的延迟几乎一样。**
resnet18 上 0.807 vs 0.852（差 5%，接近噪声），bert_tiny 上 0.296 vs 0.287
（trtexec 反而略快）。

这完全符合原理：**两条路线最终都是 TensorRT 在做优化**。前端只决定"图怎么送进去"，
不决定"kernel 怎么选"。所以：

> **不要指望换成 Torch-TensorRT 能让已经能跑的 ONNX 路线变快。它解决的是别的问题。**

它解决的问题在第三行：`customop` 上 ONNX 路线**构建失败**，而 R4 跑出了 0.202ms。

## 4.4 R4 的真正杀手锏：部分回退

`exp_unsupported_op.py` 的实测结果：

```
--- onnx_trt ---
  BUILD FAILED: onnx export failed: UnsupportedOperatorError:
    Exporting the operator 'aten::erfinv' to ONNX opset version 17 is not supported

--- dynamo_trt ---
[PASS] dynamo_trt: max_abs_err=3.662e-03, cos_sim=0.999998, top1_agree=1.000
  TRT subgraphs=2 pytorch-fallback subgraphs=1
  median=0.192ms
```

同一个模型，同一个不支持的算子 `erfinv`：

- **ONNX 路线**：在第一步就死了。`torch.onnx.export` 直接抛
  `UnsupportedOperatorError`。要继续得花工程成本：导出期写自定义 ONNX 算子或分解成
  标准算子，必要时再配一个 TensorRT plugin（第 10 章实测了这些做法）。
  **不花这个成本就是全有或全无。**
- **R4**：把图切成 3 段 —— 2 个 TRT engine（两个 `linear`）+ 1 个 PyTorch 子图
  （`erfinv`）。模型**直接就能用**，而且数值正确（`top1_agree=1.000`）。

这是本教程测到的**两条路线最重要的架构差异**。准确地说，它是"**零工程成本**能不能做"
的差异：ONNX 路线写自定义算子 + plugin 之后也能做，而且第 10 章实测比部分回退还快
（0.079 vs 0.201 ms）。另外同一个 plugin 也能注册成 Torch-TensorRT converter，
把回退子图收回 engine —— 两者不是二选一。

### 但部分回退是有代价的，代价必须量化

看完整的 `customop` 数据：

| 路线 | 中位延迟 | 相对 eager |
|---|---|---|
| eager | 0.068 ms | 1.00x |
| aoti | 0.060 ms | 1.14x |
| torch_compile | 0.135 ms | 0.50x |
| **dynamo_trt** | **0.192 ms** | **0.34x** |

**R4 在这个模型上比 eager 慢了 3 倍。**

原因就是那 3 个子图：模型总共只有 3 个算子，被切成 TRT→PyTorch→TRT 三段。
每个边界一次同步 + 一次显存往返，而每段的实际计算量小得可以忽略。
**边界开销完全主导了总时间。**

所以对 `customop` 这个模型，诚实的结论是：

> R4 让"跑不了"变成了"跑得起来但比 eager 慢"。这在工程上仍然是进步（你有了一个
> 能部署的产物），但如果你的目标是加速，这里的正确答案是 **AOTInductor（1.14x）
> 或者干脆 eager**，不是 TensorRT。

这也解释了 `min_block_size` 为什么重要：生产上你要用默认值 5 甚至更大，让它
**只在有足够大的连续可转子图时才建 engine**，碎片就留给 PyTorch。

## 4.5 R5：TorchScript 前端 —— 为什么本教程建议你不要用

R5 走 `torch.jit.trace` → `torch_tensorrt.ts.compile`。它是 Torch-TensorRT 的
老路线，官方已把 `ts` / `fx` / `ptq` 标为 **maintenance mode**。

本教程本来只打算把它当"历史兼容"列进来，但实测结果比预期糟糕得多，值得展开。

### R5 在 resnet18 上确实最快

| 路线 | resnet18 中位延迟 | 加速比 |
|---|---|---|
| **ts_trt** | **0.782 ms** | **3.68x** |
| dynamo_trt | 0.807 ms | 3.57x |
| trtexec | 0.826 ms | 3.48x |
| onnx_trt | 0.852 ms | 3.38x |

它是全表最快的。但这 3% 的优势换来了下面三个问题。

### 问题 1：bert_tiny 直接构建失败

```
bert_tiny / ts_trt (unsupported):
  RuntimeError: shape '[8, 128, 2, 64]' is invalid for input of size 0
```

`torch.jit.trace` 在处理 encoder block 里那句
`t.view(B, S, self.h, D // self.h)` 时出了问题 —— 一个"size 0"的输入说明 trace
过程中的 shape 传播就已经错了。同一个模型在 dynamo 路线上跑出 0.296ms 毫无问题。

这是 TorchScript 的典型症状：**它对 Python 的支持是一个子集，而且边界在哪不明显**，
你只能靠试。

### 问题 2：customop 上构建"成功"了，然后在运行时崩

这个更值得警惕。`exp_unsupported_op.py` 里：

```
--- ts_trt ---
  BUILD FAILED: ts_trt build failed: RuntimeError: Expected all tensors to be on
    the same device, but got mat1 is on cuda:0, different from other tensors on cpu
```

而在完整 benchmark 里（略有不同的调用序列）它走得更远，构建通过了，然后：

```
customop / ts_trt (runtime_error):
  RuntimeError: [Error thrown at core/runtime/execute_engine.cpp:112]
    Expected inputs[i].dtype() == expected_type to be true but got false
    Expected input tensors to have type Half, found type float
```

**构建期无错，运行期才崩，而且错误来自 C++ 运行时。** 根因是部分回退时
PyTorch 子图和 TRT engine 之间的 dtype 没协调好。dynamo 前端在同一个模型上
处理得很干净（`cos_sim=0.999998`）。

顺带说：正是这个失败让我发现了 `run_bench.py` 的一个真 bug —— 计时调用在
try/except 之外，所以一条"构建成功但推理失败"的路线会让整个 benchmark 进程崩掉，
后面的行全丢。修法是给推理路径单独加守卫：

```python
try:
    timing = common.bench_ms(lambda: runnable(*inputs))
except Exception as e:
    record.update(status="runtime_error", error=...)
    return record
```

**教训：设计 benchmark 时要假设"构建成功"和"推理成功"是两个独立事件。**
这条经验本身就是 R5 送的。

### 问题 3：trace 静默固化控制流

这是第 5 章的主题，这里先给结论：R5 在 `ctrlflow` 上构建成功、跑得最快
（0.059ms，1.74x），**但对没被 trace 到的那个分支输出是错的，且无任何警告**。

### R5 的结论

| | 表现 |
|---|---|
| resnet18 | 最快（3.68x），+3% over dynamo |
| bert_tiny | 构建失败 |
| customop | 运行时崩溃 |
| ctrlflow | 静默出错 |

**四个模型里三个有问题。为了 3% 的性能优势接受这个稳定性，在新项目里不划算。**
如果你有存量 TorchScript 代码在跑 R5 并且工作正常，没必要为了迁移而迁移；
但不要用它开新项目。

## 4.6 R4 的优缺点总结

优点：

- **部分回退**：不支持的算子不会让整条路线断掉，而且**零工程成本**。ONNX 路线要花成本
  写自定义算子 / plugin 才能做到（第 10 章）。
- **少一次 IR 转换**：没有 opset 版本这个约束维度，PyTorch 语义直接保真。
- **`torch.export` 的严格性**：控制流转不了就报错，不静默出错（第 5 章）。
- **返回的就是 torch Module**：不用自己写那 40 行运行时代码。

缺点：

- **依赖链最长**：torch ↔ torch-tensorrt ↔ tensorrt 三方版本要对齐（第 1 章那三个坑）。
- **没有可检查的中间产物**：ONNX 路线那个能用 Netron 打开的文件，这里没有。
  排查手段只有 `min_block_size` 调整和子图计数。
- **默认产物离不开 Python**：要脱离 Python 得另外用
  `convert_method_to_trt_engine` 或导出 `.ep`。
- **部分回退可能比 eager 慢**（`customop` 的 0.34x）。**能跑不等于该用。**
- **性能上并不比 ONNX 路线更快**（实测 5% 以内）。

## 4.7 动手

```bash
cd /root/data/learn/torch2trt/code
python3 run_bench.py --models resnet18,bert_tiny,customop --routes dynamo_trt,ts_trt
python3 exp_unsupported_op.py
```

建议自己改 `min_block_size` 跑一遍 `customop`，观察子图计数和延迟怎么变：

```python
# routes.py 里改，或者直接在 python 里
build_dynamo_trt(model, inputs, min_block_size=5)   # 什么都不转
build_dynamo_trt(model, inputs, min_block_size=1)   # 切成 3 段
```

---

← [第 3 章：ONNX → TensorRT 路线](03_onnx_trt_route.md) | [第 5 章：控制流与不支持算子](05_control_flow_and_fallback.md) →
