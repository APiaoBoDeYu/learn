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

## 4.5 R4 + 自己写的自定义算子（`torch.library`）

4.4 里的 `erfinv` 是 PyTorch 自带、只是 Torch-TensorRT 没有 converter 的 aten 算子。
实际项目里更常见的是另一种情况：**算子是你自己写的** —— 一个融合的 CUDA kernel、一个
Triton kernel —— 用 `torch.library` 注册进 PyTorch。R4 对这种算子同样有效，而且
**`build_dynamo_trt` 一行不用改**。区别只在调用它之前注册了什么。

`exp_r4_custom_op.py` 把 `erfinv` 包成自定义算子 `mylib::my_erfinv`，同一个模型、
同一份权重，测三种做法。

### 第一步（必做）：定义算子 + fake 实现

```python
@torch.library.custom_op("mylib::my_erfinv", mutates_args=())
def my_erfinv(x: torch.Tensor) -> torch.Tensor:
    return torch.erfinv(x)          # 真正的实现：CUDA / Triton / 任意 torch 代码

@my_erfinv.register_fake            # torch.export 用 fake tensor 追踪，靠它推导 shape/dtype
def _(x):
    return torch.empty_like(x)

# 模型里：x = torch.ops.mylib.my_erfinv(x)
```

`register_fake` 不能省。没有它，`torch.export` 就失败了，Torch-TensorRT 根本看不到这张图。

**只做这一步，R4 就能跑**（做法 A）：Torch-TensorRT 找不到这个算子的 converter，
就把它留在 PyTorch 里 —— 和 4.4 的部分回退完全一样。

### 做法 B：C++ plugin + 手写 converter —— 把算子收进 engine

converter 是 Torch-TensorRT 里"看到这个算子就往 TRT 网络里加什么层"的规则。
加载第 10 章那个 C++ `IPluginV3` 的 `.so`，再给 **你的算子** 注册一个 converter：

```python
import ctypes, tensorrt as trt
from torch_tensorrt.dynamo.conversion import dynamo_tensorrt_converter
from torch_tensorrt.dynamo.conversion.converter_utils import get_trt_tensor

ctypes.CDLL("artifacts/liberfinv_plugin.so")    # 加载即注册（REGISTER_TENSORRT_PLUGIN）

@dynamo_tensorrt_converter(torch.ops.mylib.my_erfinv.default, supports_dynamic_shapes=True)
def my_erfinv_converter(ctx, target, args, kwargs, name):
    creator = trt.get_plugin_registry().get_creator("ErfInv", "1", "")
    plugin = creator.create_plugin("ErfInv", trt.PluginFieldCollection([]),
                                   trt.TensorRTPhase.BUILD)
    x = get_trt_tensor(ctx, args[0], f"{name}_x")
    layer = ctx.net.add_plugin_v3([x], [], plugin)
    return layer.get_output(0)

gm, meta = routes.build_dynamo_trt(model, inputs)   # 照常调用
```

几个要对齐的地方：

- converter 的目标是 `torch.ops.<命名空间>.<算子名>.default`，即 `torch.export` 图里
  节点的 target。
- `get_creator` 的名字 / 版本要和 C++ 里 `getPluginName()` / `getPluginVersion()` 一致。
- 算子如果有标量参数（比如 `alpha`），从 `args` 里取出来，打包成 `trt.PluginField`
  放进 `PluginFieldCollection`，字段名要和 C++ 的 `getFieldNames()` 对上。

### 做法 C：Triton kernel + 自动生成 plugin —— 不写 C++

Torch-TensorRT 2.9 能从 `torch.library` 算子**自动生成** Python plugin（QDP）和
converter。前提是算子的实现本身是一个 Triton kernel 的 launch：

```python
@torch.library.custom_op("mylib::my_erfinv", mutates_args=())
def my_erfinv(x: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    erfinv_kernel[(triton.cdiv(x.numel(), 256),)](x, y, x.numel(), BLOCK=256)
    return y
# ... register_fake 同上 ...

torch_tensorrt.dynamo.conversion.plugins.custom_op(
    "mylib::my_erfinv", supports_dynamic_shapes=True)   # 一行：plugin + converter
```

### 实测

`customop` 模型（fc1 → clamp → my_erfinv → fc2），fp16，Thor，中位延迟：

| 做法 | 要写什么 | 子图 TRT / 回退 | b=32 | b=8192 | 能导出 `.plan` |
|---|---|---|---|---|---|
| A. 只定义算子（部分回退） | `custom_op` + `register_fake` | 2 / 1 | 0.228 ms | 0.362 ms | 否 |
| **B. C++ plugin + converter** | + 120 行 C++ + 10 行 converter | **1 / 0** | **0.114–0.121 ms** | **0.225 ms** | **是** |
| C. Triton + 自动生成 QDP plugin | + Triton kernel + 1 行 | 1 / 0 | 0.56–0.69 ms | 0.910 ms | 是（但运行时要 Python） |
| D. Triton + 手写 QDP AOT plugin | + Triton kernel + `trtp.register` / `aot_impl` + converter | 1 / 0 | 见下文 | — | 是，纯 TRT 可运行 |

三种做法数值都正确（cos 0.999998，`top1_agree=1.000`）。

- **B 比部分回退快 1 倍左右**，和第 10 章用 `aten.erfinv` 测的结果（0.201 → 0.118 ms）
  一致。所以算子是 aten 自带的还是你自己写的，对 converter 机制没有区别。
- **C 虽然 0 回退，却比 A 还慢 2.5 倍。** 它底层就是第 10 章测过的 QDP：在
  TRT 10.14 上每次推理都重新加载一次 PTX 模块，固定开销 0.15 ms 以上，和数据量无关。
  "子图数少"不等于"快"。C 适合验证接口和数值，不要用在热路径上。
- **只有全部算子都转进 TRT（B / C），才能导出独立的 engine。**
  `convert_exported_program_to_serialized_trt_engine` 遇到没有 converter 的算子不会回退，
  直接失败（报错是一个莫名其妙的 `UnboundLocalError: interpreter_result`）。B 导出的
  `.plan` 是 0.65 MB，用纯 TensorRT 加载、推理，结果正确。部署时**先加载 `.so` 再反序列化**，
  否则报 `IPluginCreatorInterface not found`（附录 07）。
- **C 导出的 `.plan` 不能独立部署。** 换一个新进程用纯 TensorRT 加载，可以反序列化，
  但创建执行上下文时报 `Cannot find plugin: my_erfinvmylib`。原因是自动生成的只是 JIT
  插件：`.plan` 里只记录插件名，执行时由 TRT 回调 Python 函数，再调用
  `torch.ops.mylib.my_erfinv` 启动 Triton kernel。

### 做法 D：手写 QDP AOT —— 把 Triton kernel 嵌进 engine

`plugins.custom_op` 不会生成 AOT 实现，要自己用 `tensorrt.plugin` 写：`trtp.register`
声明输出形状，`trtp.aot_impl` 在构建 engine 时用 Triton 把 kernel 编译成 PTX，并返回
PTX 和启动参数；converter 里用 `ctx.net.add_plugin(trtp.op.<ns>.<name>(x), aot=True)`。
这时 PTX 会被写进 `.plan`，推理时不需要 Python、Triton，也不需要 `.so`。实验代码不在本仓库，
放在 `/tmp/aot/`。

实测（b=32，纯 TRT `execute_async_v3` 的中位延迟）：

| | 新进程纯 TRT 运行 | 纯 TRT 执行耗时 |
|---|---|---|
| B. C++ plugin | 成功（先加载 `.so`） | **0.029 ms** |
| D. QDP AOT（PTX） | 成功，cos 0.999998 | 0.214 ms |
| D'. QDP AOT 改为返回 cubin | 失败：`onShapeChange status -1`，输出 nan | — |

踩过的坑：

- Triton 3.5 编译出的 kernel 会在参数末尾多加两个 scratch 指针，TRT 只传 `x, n, y`，
  参数个数对不上，启动就段错误。本例的 scratch 大小为 0，可以把这两个参数从 PTX 里删掉
  （这是针对当前版本的变通做法）。
- Triton kernel 里不能用 Python 列表写 `for c in [...]` 循环，否则编译失败。
- `aot_impl` 目前只接受 PTX。无论是 Triton 直接生成的 cubin，还是用 `ptxas` 编译出的 cubin，
  构建都能通过，但推理时都会失败。
- **慢的原因**：用 `nsys` 统计，D 每推理一次就调用一次 `cuModuleLoadData`，把 PTX 模块
  重新加载一遍（推理 220 次，调用 223 次；B 全程只调用 3 次）。关掉驱动的 JIT 缓存
  （`CUDA_CACHE_DISABLE=1`）后，每次推理要 10～16 ms，说明每次加载都要重新 JIT 编译 PTX。

### 怎么选

**结论：R4（torch.export → Torch-TensorRT dynamo_trt）遇到不支持的算子时，最优做法是
写 C++ plugin + converter（做法 B）。** 实测四种做法中只有它同时做到了：最快、
整张图进 engine、导出的 `.plan` 能脱离 Python 用纯 TensorRT 部署，而且同一个 `.so`
还能给 ONNX 路线用。C 运行时离不开 Python；D 虽然能独立部署，但在 TRT 10.14 上
每次推理都要重新加载 PTX，比 B 慢约 7 倍。

```
1. 定义 custom_op + register_fake，用 R4 跑通      零额外成本，部分回退
   └─ profile 证明回退子图是瓶颈 / 需要脱离 Python ↓
2. 写 C++ IPluginV3 + converter                   最优：最快，可导出 .plan，.so 同时能给 ONNX 路线用
（做法 C 只用来快速验证接口和数值；做法 D 能脱离 Python，但 TRT 10.14 上每次推理都重新加载 PTX，别上热路径）
```

这也补全了 4.4 的结论：**部分回退和 plugin 不是二选一**。部分回退让你先拿到一个
能用的产物，plugin 是确认瓶颈之后再把回退的那段收回 engine。

## 4.6 R5：TorchScript 前端 —— 为什么本教程建议你不要用

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

## 4.7 R4 的优缺点总结

优点：

- **部分回退**：不支持的算子不会让整条路线断掉，而且**零工程成本**。ONNX 路线要花成本
  写自定义算子 / plugin 才能做到（第 10 章）。
- **自定义算子是一等公民**：`torch.library` 算子 + `register_fake` 就能跑；
  再注册一个 converter 就能把它收进 engine，甚至导出 `.plan`（4.5）。
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

## 4.8 动手

```bash
cd /root/data/learn/torch2trt/code
python3 run_bench.py --models resnet18,bert_tiny,customop --routes dynamo_trt,ts_trt
python3 exp_unsupported_op.py
bash plugins/build.sh && python3 exp_r4_custom_op.py   # 4.5：R4 + torch.library 自定义算子
```

建议自己改 `min_block_size` 跑一遍 `customop`，观察子图计数和延迟怎么变：

```python
# routes.py 里改，或者直接在 python 里
build_dynamo_trt(model, inputs, min_block_size=5)   # 什么都不转
build_dynamo_trt(model, inputs, min_block_size=1)   # 切成 3 段
```

---

← [第 3 章：ONNX → TensorRT 路线](03_onnx_trt_route.md) | [第 5 章：控制流与不支持算子](05_control_flow_and_fallback.md) →
