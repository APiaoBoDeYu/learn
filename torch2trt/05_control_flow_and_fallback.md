# 第 5 章：控制流与不支持算子 —— 本教程最重要的一章

前面几章比的是速度。本章比的是**正确性**，而这一章的结论应该改变你的工作流程。

核心发现先放最前面：

> **在含数据相关控制流的模型上，`torch.onnx.export` 和 `torch.jit.trace` 会
> 静默产出一个错误的模型。构建成功、无任何警告、跑得还最快 —— 但对一半的输入，
> 输出是错的。**

## 5.1 实验设计

`models.py` 里的 `DataDependentBranch`：

```python
def forward(self, x):
    if x.mean() > 0:
        x = F.relu(self.a(x))
    else:
        x = torch.tanh(self.b(x))
    return self.out(x)
```

`x.mean() > 0` 依赖**输入的值**，不是依赖配置。导出期无法确定走哪个分支。

怎么测（`exp_control_flow.py`）：

1. 造两个输入：`pos_in = x.abs()`（均值 > 0，走 relu 分支）和
   `neg_in = -x.abs()`（均值 < 0，走 tanh 分支）。
2. 用 `pos_in` 作为示例输入去构建每条路线 —— **只给它看正分支**。
3. 然后用**两个**输入分别推理，各自和 eager 的对应输出比。

关键是第 3 步测了"没给它看过的那个分支"。这是唯一能发现静默固化的方法。

实验还带一个**有效性自检**，防止测出一个空结论：

```python
branch_gap = cosine_similarity(ref_pos, ref_neg)
print(f"cos(eager pos-branch, eager neg-branch) = {branch_gap:.5f}"
      "   (must be well below 1, or the test is vacuous)")
```

如果两个分支本来就输出差不多，那这个实验什么都证明不了。必须先确认它们确实不同。

## 5.2 实测结果

```
### control-flow verdict

| route | outcome | cos(traced branch) | cos(other branch) |
|---|---|---|---|
| eager | correct_both_branches | 1.00000 | 1.00000 |
| torch_compile | correct_both_branches | 1.00000 | 1.00000 |
| ts_trt | SILENTLY_WRONG_on_untraced_branch | 1.00000 | 0.38649 |
| dynamo_trt | refused | - | - |
| onnx_trt | SILENTLY_WRONG_on_untraced_branch | 1.00000 | 0.38649 |
| aoti | refused | - | - |
```

三种截然不同的行为：

### 行为 A：静默出错（onnx_trt、ts_trt）

`cos(traced branch) = 1.00000` —— 被 trace 的那个分支完美。
`cos(other branch) = 0.38649` —— 另一个分支**几乎毫无关系**。

注意 0.386 这个数字：它不是"精度略有损失"，它是"算的是另一个函数"。
trace 把 `relu(self.a(x))` 这个分支烧进了图里，`self.b`、`tanh` 和那个 `if`
在产物里**根本不存在**。对负均值输入，engine 照样跑 relu 分支，给你一个
语法上合法、语义上错误的结果。

**而且构建过程零警告。** 这是最危险的部分 —— 没有任何信号提示你产物是残缺的。

再看这两条路线在这个模型上的速度：

| 路线 | ctrlflow 延迟 | 加速比 |
|---|---|---|
| ts_trt | 0.059 ms | **1.74x** |
| onnx_trt | 0.104 ms | 0.98x |
| eager | 0.102 ms | 1.00x |

`ts_trt` 是这个模型上**最快**的路线。当然快 —— 它少算了一整个分支的可能性。

> **一个 benchmark 如果只测延迟，会把这条错误的路线评为冠军。**
> 这就是第 2 章那些方法论纪律存在的理由。

### 行为 B：诚实拒绝（dynamo_trt、aoti）

两条基于 `torch.export` 的路线都拒绝构建：

```
GuardOnDataDependentSymNode: Could not guard on data-dependent expression
Eq(u0, 1) (unhinted: Eq(u0, 1)).  (Size-like symbols: none)
```

`torch.export` 还很贴心地打出了它抓到的图，直接指到源码行：

```python
# File: models.py:129 in forward, code: if x.mean() > 0:
mean: "f16[]" = torch.ops.aten.mean.default(arg6_1)
gt:   "b8[]"  = torch.ops.aten.gt.Scalar(mean, 0)
ne:   "b8[]"  = torch.ops.aten.ne.Scalar(gt, 0)
item: "Sym(Eq(u0, 1))" = torch.ops.aten.item.default(ne)
```

读一下这段，`torch.export` 的机制就很清楚了：它把 `if` 条件的求值过程照实记录成
`mean → gt → ne → item`，走到 `item()` 时需要把一个 tensor 变成 Python bool
才能决定分支。那个值只有运行时才知道，于是它创造了一个符号 `u0` 并要求给出一个
guard —— 给不出，就报错。

**这个报错是好事。** 它是 `torch.export` 相对 trace 的核心设计取舍：
**宁可拒绝，也不静默产出错误的东西。**

### 行为 C：graph break，保持正确（torch_compile）

`torch.compile` 两个分支都是 1.00000。机制：Dynamo 在 `if` 处切一刀
（graph break），把 `if` 留给 CPython 执行，两侧各编译成一张图。
运行时哪个分支都能走，因为判断真的在运行时做。

代价在延迟上：

| 路线 | ctrlflow 延迟 | 加速比 |
|---|---|---|
| eager | 0.102 ms | 1.00x |
| torch_compile | 0.258 ms | **0.40x** |

**torch.compile 在这个模型上比 eager 慢 2.5 倍。** 模型太小（0.53M 参数），
graph break 之后每段图里几乎没有可融合的东西，而 guard 检查 + 多段图的调度开销
是实打实的。

所以 `torch.compile` 在这里是"正确但更慢"。这也是个诚实的结论：
**它的正确性保证不是免费的。**

## 5.3 这对你的工作流意味着什么

如果你现在在跑 ONNX 路线，请做这件事：

> **检查你的模型里有没有依赖 tensor 值的 Python 控制流。**
> 有的话，你现在的 engine 很可能对一部分输入是错的，而你的测试如果只用了
> 一批同分布的样本，是发现不了的。

具体怎么查：

**方法 1（推荐）：拿 `torch.export` 当 linter。** 不用真的迁移路线，只是试着导一次：

```python
try:
    torch.export.export(model, example_inputs)
    print("no data-dependent control flow")
except Exception as e:
    print("FOUND:", e)      # 它会指到具体源码行
```

`torch.export` 的严格性在这里变成一个免费的静态检查工具。**这可能是本章最实用的
一条建议**：即使你坚持用 ONNX 路线，也应该在 CI 里加这一步。

**方法 2：用双输入测试。** 就是 `exp_control_flow.py` 的做法 —— 拿刻意走不同分支的
输入去测 engine，不要只用示例输入那一类。

**方法 3：`torch.onnx.export(..., dynamo=True)`。** 走 `torch.export` 抓图再转 ONNX，
于是也继承了严格性。代价是一部分老模型导不出来了 —— 但那些"导不出来"的里面，
有些本来就是导出了也是错的。

## 5.4 怎么真正修好（而不是绕过）

如果模型确实需要这个分支，三个选择：

**选择 1：把控制流移出模型。** 最简单，通常也最对。在模型外面判断，调两个不同的
（各自可导出的）子模型。部署上多一点胶水代码，但每个产物都是完整正确的。

**选择 2：用 `torch.cond`。** PyTorch 提供的可导出控制流原语：

```python
import torch
x = torch.cond(pred, true_fn, false_fn, (x,))
```

`torch.export` 能把它表达成图里的一个 `cond` 节点，两个分支都保留。
注意后端要支持 —— TensorRT 对 `cond` 的支持有限，很可能落到 PyTorch 回退子图里。

**选择 3：接受 `torch.compile` 的 graph break。** 正确、零改造，代价是本章测到的
性能损失（这个小模型上 0.40x；模型越大、graph break 之外的部分越多，代价越小）。

graph break 的本质不是“报错”，而是“编译边界被切开了”。它是 torch.compile 为了兼容任意 Python 代码而做的降级机制，但过多的 graph break 往往会让性能变差，因为大图被切碎后，跨算子优化、fusion、调度优化都会受影响。

## 5.5 第二个实验：不支持的算子

`exp_unsupported_op.py`，模型 `customop`（fc1 → clamp → erfinv → fc2）含 `torch.erfinv`：
正常的 aten 算子，但**没有 ONNX opset 映射，也没有 TensorRT converter**。

只测"不做任何额外工作"的结果是不够的：对 ONNX 路线来说，遇到不支持的算子时的
**标准做法**是"导出侧注册自定义 ONNX 算子 + 构建侧提供同名 TensorRT plugin"。
所以本实验对 `onnx_trt`（Python builder）和 `trtexec`（CLI）既测了什么都不做，
也测了两侧都修好的情况；另外用 trtexec 单独测了"只修导出侧"的中间状态，把两个
失败点分开展示。

```bash
cd code
bash plugins/build.sh            # 编译 C++ IPluginV3 → artifacts/liberfinv_plugin.so
python3 exp_unsupported_op.py    # 日志：artifacts/unsupported_op.log
```

### 实测结果（Thor，FP16，b=32）

| route | 做了什么 | outcome | median ms | trtexec GPU ms | cos | TRT / 回退子图 |
|---|---|---|---|---|---|---|
| eager | — | 能跑 | 0.065 | — | 0.999999 | — |
| torch_compile | — | 能跑 | 0.132 | — | 1.000000 | — |
| onnx_trt | 什么都不做 | **导出失败** | — | — | — | — |
| trtexec | 什么都不做 | **导出失败** | — | — | — | — |
| trtexec | 只注册自定义 ONNX 算子 | **parse 失败** | — | — | — | — |
| **onnx_trt** | **自定义 ONNX 算子 + C++ plugin** | **能跑** | **0.092** | **0.022** | 0.999998 | 1 / 0 |
| **trtexec** | **自定义 ONNX 算子 + C++ plugin** | **能跑** | **0.090** | **0.021** | 0.999998 | 1 / 0 |
| dynamo_trt | 什么都不做 | 能跑（部分回退） | 0.193 | — | 0.999998 | 2 / 1 |
| ts_trt | 什么都不做 | 构建成功，**运行时崩溃** | — | — | — | — |

- **median ms**：和第 6 章同一套 CUDA event 计时，含 Python 调用开销（TRT 路线走第 3 章的 `TrtRunner`）。
- **trtexec GPU ms**：同一个 `.plan` 用 `trtexec --loadEngine` 加载，只算 GPU 时间。
  这一列有值，本身就证明 engine 可以在**没有 Python 的 C++ 进程**里加载运行。

### ONNX 路线：两个失败点，对应两个修复点

ONNX 路线有两步，`erfinv` 在每一步都会卡一次，修复也正好是两处：

```
nn.Module ──[torch.onnx.export]──> ONNX ──[TRT OnnxParser + Builder]──> .plan
             ① 卡在这：没有 symbolic         ② 卡在这：不认识的 op_type
             修：注册自定义 ONNX 算子           修：提供同名 TensorRT plugin
```

**① 什么都不做 —— 死在 PyTorch 侧。**

```
UnsupportedOperatorError: Exporting the operator 'aten::erfinv' to ONNX
opset version 17 is not supported
```

连 ONNX 文件都没生成，`onnx_trt` 和 `trtexec` 一起倒（trtexec 吃的就是这个 ONNX）。
所以这一步你怪不到 TensorRT 头上。报错里打出的 Torch IR 图中，`aten::erfinv`
那一行一眼可见 —— 它就是你要写 symbolic 的那个算子。

**修复 ①：注册自定义 ONNX 算子。** 告诉导出器"遇到 `aten::erfinv` 就生成一个
`custom::ErfInv` 节点"：

```python
def erfinv_symbolic(g, x):
    return g.op("custom::ErfInv", x).setType(x.type())

torch.onnx.register_custom_op_symbolic("aten::erfinv", erfinv_symbolic, 17)
torch.onnx.export(model, inputs, "customop_plugin.onnx", opset_version=17,
                  custom_opsets={"custom": 1}, dynamo=False)
```

导出成功，图里是 `Gemm → Clip → custom::ErfInv → Gemm`。

**② 只修 ① —— 死在 TensorRT parser。**

```
$ trtexec --onnx=customop_plugin.onnx --fp16
[E] [TRT] ERROR: onnxOpCheckers.cpp:884 In function checkFallbackPluginImporter:
[6] creator && "Plugin not found, are the plugin name, version, and namespace correct?"
```

`checkFallbackPluginImporter` 这个名字把机制说清楚了：parser 遇到不认识的 op_type
**不会直接报错**，而是回退到"按名字（`ErfInv`，version `"1"`，namespace `""`）去
plugin registry 里找 creator"，找不到才失败。所以两侧对接靠的就是**一个字符串**：
ONNX 节点的 op_type = plugin 的名字。

**修复 ②：提供 TensorRT plugin。** `code/plugins/erfinv_plugin.cu` 是一个 IPluginV3
实现（kernel 10 行，其余约 110 行是 shape/dtype/序列化/creator 样板，详见第 10 章），
编成 `liberfinv_plugin.so`。`.so` 被加载时 `REGISTER_TENSORRT_PLUGIN` 的静态构造
函数就把 creator 注册进全局 registry，两种构建方式只是加载方式不同：

```python
ctypes.CDLL("artifacts/liberfinv_plugin.so")   # Python builder：parse 前加载即可
```
```bash
trtexec --onnx=customop_plugin.onnx --fp16 \
        --staticPlugins=artifacts/liberfinv_plugin.so   # CLI
```

两者都构建成功，network 里 `/ErfInv` 是一个 PLUGIN_V3 层，engine 是**一个完整的
TensorRT engine，没有任何回退子图**。数值正确（cos 0.999998）。

**部署时别忘了 `.so`。** engine 里只记了"这里有个叫 ErfInv 的 plugin"，代码不在里面。
不带 `--staticPlugins` 去加载：

```
$ trtexec --loadEngine=customop_plugin_cli.plan
[E] [TRT] IPluginRegistry::getCreator: Error Code 4: API Usage Error
    (Cannot find plugin: ErfInv, version: 1, namespace:.)
[E] Error[1]: ICudaEngine::createExecutionContext: ... Cannot deserialize plugin
    since corresponding IPluginCreatorInterface not found in Plugin Registry
```

注意它在 `createExecutionContext` 才失败，不是在反序列化时 —— 健康检查只做
`deserialize_cuda_engine` 是查不出来的。另外 `.so` 按 `-arch=sm_110a` 编译，换 GPU
架构要重编。

### 其他路线

**`dynamo_trt` —— 什么都不做也能跑，但切成 3 段。** `TRT subgraphs=2, fallback=1`：
两个 `linear` 各自进了 engine，`erfinv` 留在 PyTorch。数值正确，但 0.193 ms 比
eager 的 0.065 ms **慢了 3 倍**：3 个算子的模型切 3 段，边界的同步和显存往返完全压过
了计算本身（上一章解释过）。同一个 `.so` 也能注册成 Torch-TensorRT converter，
让它变成 1 段、0 回退，0.118 ms —— 见第 10 章。

**`ts_trt` —— 构建成功，第一次推理就崩。**

```
RuntimeError: [Error thrown at core/runtime/execute_engine.cpp:112]
Expected input tensors to have type Half, found type float
```

部分回退时，回退出来的 PyTorch 段和 TRT engine 之间 dtype 没协调好（详见上一章）。
这比构建失败更糟：构建期一切正常，问题拖到了运行期。

**`torch_compile` —— 毫无问题。** Inductor 直接为 `erfinv` 生成 kernel，没有
"不支持"这个概念。比 eager 慢（0.132 vs 0.065）纯粹是因为模型太小，编译收益盖不住开销。

### 结论：plugin 让 ONNX 路线能跑；快不快取决于你怎么调用 engine

| 路线 | 额外工作 | customop 延迟 | 加速比 | 能脱离 Python 部署吗 |
|---|---|---|---|---|
| eager | — | 0.063 ms | 1.00x | 否 |
| aoti | 无 | 0.055 ms | 1.15x | **是**（`.pt2`） |
| onnx_trt + plugin（TrtRunner） | 自定义 ONNX 算子 + C++ plugin | 0.105 ms | 0.60x | **是**（`.plan` + `.so`） |
| trtexec + plugin（TrtRunner） | 同上 | 0.094 ms | 0.67x | **是**（`.plan` + `.so`） |
| trtexec + plugin（精简封装） | 同上 + 一个像样的封装 | **0.034 ms** | **1.85x** | **是** |
| trtexec + plugin（CUDA graph） | 同上 + CUDA graph 回放 | **0.022 ms** | **2.86x** | **是** |
| torch_compile | 无 | 0.132 ms | 0.48x | 否 |
| dynamo_trt | 无 | 0.193 ms | 0.33x | 否 |
| onnx_trt / trtexec | 无 | 导出失败 | — | — |
| ts_trt | 无 | 运行时崩溃 | — | — |

（前六行来自下文的 `exp_plugin_latency.py`，在同一个进程里测得；torch_compile 和 dynamo_trt
两行来自上面 `exp_unsupported_op.py` 那次运行，加速比按 eager 0.063 ms 折算。这么短的延迟
每次运行有几 µs 的波动，aoti 三次分别测得 0.068 / 0.054 / 0.055 ms。）

三点：

1. **对 ONNX 路线，"不支持的算子"不是死路，是工程量。** 一个 4 行的 symbolic
   加一个 120 行的 C++ plugin，`onnx_trt` 和 `trtexec` 都从"构建失败"变成"一个完整
   的 engine，零回退，能被不带 Python 的 trtexec 直接加载"。前一版把这两行记成
   "构建失败"，只说明了"不做额外工作"时的情况。
2. **plugin 修好的是"能不能"；表里 ONNX + plugin 比 eager 慢，原因不在 engine，在调用它的
   Python 代码** —— 见下一节。
3. **写 plugin 之前先想想能不能不写。** 在导出期把 `erfinv` 分解成标准 ONNX 算子，
   不需要 plugin、不用带 `.so`，而且 TensorRT 能把它和相邻层融合（plugin 是融合屏障）
   —— 第 10 章实测这比 C++ plugin 更快。

### 为什么 `trtexec + plugin` 比 eager 还慢？

先澄清一点：`trtexec` 只是**构建** engine 的方式。表里 0.090 ms 的那一列，是把
trtexec 产出的 `.plan` 加载进 Python、用第 3 章的 `TrtRunner` 调用测出来的；
`onnx_trt + plugin` 是同一张网络，所以两者延迟一样。

`exp_plugin_latency.py` 专门拆解了这件事（日志 `artifacts/plugin_latency.log`）：

```bash
cd code
python3 exp_plugin_latency.py                                   # 计时 + TrtRunner 拆解
nsys profile -t cuda -o /tmp/r python3 exp_plugin_latency.py --nsys cli   # kernel 时间
nsys stats -r cuda_gpu_kern_sum /tmp/r.nsys-rep                 # 另有 eager / aoti / py
```

**第一步：GPU 上真正干活的时间（nsys，1000 次平均）—— TRT 并不慢。**

| 路线 | kernel | 每次 GPU 时间 |
|---|---|---|
| eager | GEMM 4.2 µs + GEMM 2.6 µs + clamp 1.3 µs + erfinv 1.3 µs（4 个） | 9.4 µs |
| aoti | GEMM 4.3 µs + GEMM 2.5 µs + **clamp+erfinv 融合** 1.3 µs（3 个） | 8.1 µs |
| trtexec + plugin | GEMM 5.0 µs + **GEMM+bias+clamp 融合** 3.7 µs + plugin erfinv 1.2 µs（3 个） | 9.8 µs |

三条路线的 GPU 时间都在 8–10 µs，**谁都不比谁快多少**：32×512 的矩阵乘太小，喂不饱
GPU，TensorRT 能做的优化（融合 clamp、选 tactic）在这个量级上省不出东西。
所以端到端快慢**完全由 host 侧开销决定**，而延迟（50–100 µs）比 GPU 时间大 5–10 倍。

**第二步：重新测一遍四个方案，并把 TrtRunner 换掉（b=32，CUDA event median，跑两次结果一致）：**

| 方案 | event ms | wall ms | 说明 |
|---|---|---|---|
| eager | 0.063 | 0.060 | 每个算子一次 Python 分发 + kernel launch |
| aoti | 0.055 | 0.054 | 一次调用跑完整张图，少一个 kernel |
| onnx_trt + plugin（TrtRunner） | 0.105 | 0.101 | 第 5 章表里的数 |
| trtexec + plugin（TrtRunner） | 0.094 | 0.090 | 同上，同一网络 |
| trtexec + plugin（精简封装） | **0.034** | **0.038** | 同一个 `.plan`，只改 Python 调用代码 |
| trtexec + plugin（CUDA graph） | **0.022** | 0.043 | 同上 + CUDA graph 回放 |
| *参考：`trtexec --loadEngine`（C++）* | *0.028 host latency / 0.021 GPU* | — | *加 `--useCudaGraph`：GPU 0.013* |

**同一个 engine，换个 runner 就从 0.094 ms 变成 0.034 ms，比 eager 快将近一倍。**
慢的是 `TrtRunner`，不是 TensorRT。

**第三步：TrtRunner 的时间花在哪了。** 逐项去掉再测（host 侧每次调用耗时）：

| TrtRunner 每次调用都在做的事 | 增加的 host 时间 |
|---|---|
| 基线：`set_input_shape` + `set_tensor_address` + `execute_async_v3`（当前 stream） | 0.019 ms |
| `torch.empty` 每次新分配输出 + `get_tensor_shape` | +0.011 ms |
| `_torch_dtype` 每次调用都**重建一个 dict** 查 dtype（每次调用 2 次，各 4.4 µs） | +0.010 ms |
| **用自己的 `torch.cuda.Stream` 再 `stream.synchronize()`** | **+0.031 ms** |
| 合计 ≈ 完整 TrtRunner | 0.074 ms |

最大的一项是**自建 stream + 同步**：TrtRunner 在单独的 stream 上 enqueue，然后在
`__call__` 里阻塞等它跑完才返回。这样 host 没法在 GPU 执行时提前准备下一次调用，
GPU 的 9.8 µs 被**串行**加到了 host 开销上。eager 和 aoti 都在当前 stream 上异步提交，
CUDA event 计时里 host 和 GPU 是重叠的。

（这也暴露了一个计时口径问题：`common.bench_ms` 在当前 stream 上打 event，而
TrtRunner 的 kernel 跑在另一个 stream 上 —— 它能测出"对"的数，只是因为 TrtRunner
在 end event 之前就同步了。）

精简封装只做三件事：**输入/输出 buffer 预分配并在构造时绑定地址、shape 固定时
不重复 set、在调用方的 stream 上 enqueue 且不同步**（见 `exp_plugin_latency.py`
的 `LeanRunner`），每次调用只剩一次 `copy_` 和一次 `execute_async_v3`。

**结论更新：**

- "ONNX + plugin 比 eager 慢"是**测量对象选错了**：测的是 40 行教学用的 TrtRunner，
  不是 TensorRT。真正部署时（C++，或者精简的 Python 封装），engine 的
  0.021–0.034 ms 比 eager 的 0.063 ms 快。
- 但这个优势**不是 TensorRT 优化出来的** —— GPU 时间三者一样（8–10 µs）。它来自"一次调用
  跑完整张图、host 开销小"。**小模型上，比的是 host 开销，不是编译器**（第 6 章规律 1）。
- 跟 AOTI 比：默认用法下 AOTI（0.055）比精简封装（0.034）慢，因为 AOTI 的 Python
  调用（`aoti_load_package` 返回的包装器）也有自己的开销；两者 GPU 时间只差 1.7 µs
  （8.1 vs 9.8 µs）。本次没有测 AOTI 在 C++ 里的用法，
  所以不能说 TensorRT 在这里就赢了 AOTI。第 10 章 b=8192 计算受限的情形下 AOTI 仍更快
  （0.144 vs 0.233 ms，GPU 时间也是 AOTI 占优）。

**不做额外工作时，唯一既比 eager 快、又能脱离 Python 部署的是 AOTInductor ——
一条完全不碰 TensorRT 的路线。** 愿意做额外工作（plugin + 一个像样的封装），
ONNX + plugin 也能脱离 Python 部署，在这个小模型上也能跑赢默认用法的 eager 和 AOTI ——
但赢在 host 开销，不在 GPU。

这是整个教程里最反直觉的结论之一：如果只按"TensorRT 是 NVIDIA 推理的标准答案"来选，
又只用零成本的部分回退（`dynamo_trt`），你会选到一个比什么都不做还慢 3 倍的方案；
而如果用一个随手写的 runner 去测 TensorRT，又会把它冤枉成"比 eager 还慢"。

## 5.6 本章要记住的三件事

1. **trace 类路线（ONNX 默认、TorchScript）在数据相关控制流上会静默出错**，
   而且可能在延迟表上排第一。用 `torch.export` 当 linter 去查。
2. **`torch.export` 的报错是特性不是缺陷。** 它拒绝的东西，trace 会照做但做错。
3. **不支持的算子：ONNX 路线靠"自定义 ONNX 算子 + TensorRT plugin"修，但"能转"不等于"该转"。**
   两侧靠一个字符串（op_type = plugin 名）对接，修好后 `onnx_trt` / `trtexec` 都能产出
   零回退的完整 engine；`dynamo_trt` 零成本能跑，但切段后比 eager 慢 3 倍。
   子图计数、GPU 时间和 host 开销都要分开看，才知道值不值得 —— 小模型上 TrtRunner
   一类的调用开销能让一个更快的 engine 看起来比 eager 还慢。

---

← [第 4 章：Torch-TensorRT 路线](04_torch_tensorrt_route.md) | [第 6 章：综合对比与选型](06_benchmark_comparison.md) →
