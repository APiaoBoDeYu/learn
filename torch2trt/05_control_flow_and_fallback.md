# 第 5 章：控制流与不支持算子

本章比的是**正确性**，不是速度。核心结论：

> **在含数据相关控制流的模型上，`torch.onnx.export`（默认）和 `torch.jit.trace` 会
> 静默产出错误的模型：构建成功、零警告、延迟还可能最低，但对没被 trace 到的分支输出是错的。**

## 5.1 实验：数据相关分支

`models.py` 的 `DataDependentBranch`：

```python
def forward(self, x):
    if x.mean() > 0:
        x = F.relu(self.a(x))
    else:
        x = torch.tanh(self.b(x))
    return self.out(x)
```

分支取决于**输入的值**，导出期无法确定。`exp_control_flow.py` 的测法：

1. 构造 `pos_in = x.abs()`（走 relu）和 `neg_in = -x.abs()`（走 tanh）。
2. 只用 `pos_in` 构建各路线。
3. 用**两个**输入分别推理并与 eager 对比 —— 测"没见过的分支"是发现静默固化的唯一办法。
4. 自检：`cos(eager pos, eager neg)` 必须远小于 1，否则两个分支本就相近，实验无效。

## 5.2 结果：三种行为

| route | outcome | cos(traced branch) | cos(other branch) | 延迟 / 加速比 |
|---|---|---|---|---|
| eager | 正确 | 1.00000 | 1.00000 | 0.102 ms / 1.00x |
| torch_compile | 正确（graph break） | 1.00000 | 1.00000 | 0.258 ms / 0.40x |
| ts_trt | **静默出错** | 1.00000 | 0.38649 | 0.059 ms / **1.74x** |
| onnx_trt | **静默出错** | 1.00000 | 0.38649 | 0.104 ms / 0.98x |
| dynamo_trt | 拒绝构建 | - | - | - |
| aoti | 拒绝构建 | - | - | - |

**A. 静默出错（onnx_trt、ts_trt）。** trace 把 relu 分支烧进图里，`self.b`、`tanh` 和 `if`
在产物中根本不存在。cos 0.386 不是精度损失，而是"算的是另一个函数"。构建过程零警告。
`ts_trt` 还是最快的路线 —— **只测延迟的 benchmark 会把错误的路线评为冠军。**

**B. 诚实拒绝（dynamo_trt、aoti，均基于 `torch.export`）。**

```
GuardOnDataDependentSymNode: Could not guard on data-dependent expression Eq(u0, 1)
# File: models.py:129 in forward, code: if x.mean() > 0:
```

`torch.export` 把条件记录为 `mean → gt → ne → item`；`item()` 需要把 tensor 转成
Python bool，值只在运行时可知，于是生成符号 `u0` 并要求 guard，给不出就报错，且指到源码行。
这是 `torch.export` 的设计取舍：**宁可拒绝，也不静默出错。**

**C. graph break，保持正确（torch_compile）。** Dynamo 在 `if` 处切开图，`if` 交给 CPython
在运行时判断，两侧分别编译。代价是性能：模型很小（0.53M 参数），切段后几乎没有可融合的东西，
guard 检查和多段调度开销使其比 eager 慢 2.5 倍。graph break 不是报错，而是编译边界被切开；
过多的 graph break 会破坏融合和调度优化。

## 5.3 如何检查与修复

**检查**（如果你在用 ONNX 路线）：

1. **拿 `torch.export` 当 linter（推荐，建议放进 CI）：**
   ```python
   try:
       torch.export.export(model, example_inputs)
   except Exception as e:
       print("FOUND:", e)   # 指到具体源码行
   ```
2. **双输入测试：** 用刻意走不同分支的输入测 engine，不要只用示例输入同分布的样本。
3. **`torch.onnx.export(..., dynamo=True)`：** 基于 `torch.export` 抓图，继承其严格性；
   部分老模型会导不出来，但其中一些本来导出了也是错的。

**修复**（模型确实需要分支时）：

1. **把控制流移出模型（通常最好）：** 在外部判断，调用两个各自可导出的子模型。
2. **`torch.cond(pred, true_fn, false_fn, (x,))`：** 可导出的控制流原语，两个分支都保留在图中；
   但 TensorRT 对 `cond` 支持有限，很可能落入 PyTorch 回退子图。
3. **接受 `torch.compile` 的 graph break：** 正确、零改造；模型越大、break 外的计算越多，代价越小。

## 5.4 实验：不支持的算子

模型 `customop`（fc1 → clamp → erfinv → fc2）。`torch.erfinv` 是普通 aten 算子，但
**没有 ONNX opset 映射，也没有 TensorRT converter**。

```bash
cd code
bash plugins/build.sh            # 编译 C++ IPluginV3 → artifacts/liberfinv_plugin.so
python3 exp_unsupported_op.py    # 日志：artifacts/unsupported_op.log
```

结果（Thor，FP16，b=32）：

| route | 额外工作 | outcome | median ms | cos | TRT / 回退子图 |
|---|---|---|---|---|---|
| eager | — | 能跑 | 0.065 | 0.999999 | — |
| torch_compile | — | 能跑 | 0.132 | 1.000000 | — |
| onnx_trt / trtexec | 无 | **导出失败** | — | — | — |
| trtexec | 仅自定义 ONNX 算子 | **parse 失败** | — | — | — |
| onnx_trt / trtexec | 自定义 ONNX 算子 + C++ plugin | 能跑 | 0.09（GPU 0.021） | 0.999998 | 1 / 0 |
| dynamo_trt | 无 | 能跑（部分回退） | 0.193 | 0.999998 | 2 / 1 |
| ts_trt | 无 | 构建成功，**运行时崩溃** | — | — | — |

### ONNX 路线：两个失败点，两个修复点

```
nn.Module ──[torch.onnx.export]──> ONNX ──[TRT OnnxParser + Builder]──> .plan
             ① 没有 symbolic              ② 不认识的 op_type
             修：注册自定义 ONNX 算子        修：提供同名 TensorRT plugin
```

**① 导出失败（PyTorch 侧，与 TensorRT 无关）：**
`UnsupportedOperatorError: Exporting the operator 'aten::erfinv' to ONNX opset version 17 is not supported`

修复：注册 symbolic，把 `aten::erfinv` 导出为 `custom::ErfInv` 节点：

```python
def erfinv_symbolic(g, x):
    return g.op("custom::ErfInv", x).setType(x.type())

torch.onnx.register_custom_op_symbolic("aten::erfinv", erfinv_symbolic, 17)
torch.onnx.export(model, inputs, "customop_plugin.onnx", opset_version=17,
                  custom_opsets={"custom": 1}, dynamo=False)
```

**② parse 失败（TensorRT 侧）：**
`checkFallbackPluginImporter: Plugin not found, are the plugin name, version, and namespace correct?`

parser 遇到不认识的 op_type 时，会按名字（`ErfInv`、version `"1"`、namespace `""`）去 plugin
registry 查找 creator，找不到才失败。**两侧靠一个字符串对接：ONNX op_type = plugin 名。**

修复：`code/plugins/erfinv_plugin.cu`（IPluginV3，kernel 约 10 行，其余为样板，详见第 10 章）。
`.so` 加载时由 `REGISTER_TENSORRT_PLUGIN` 自动注册 creator：

```python
ctypes.CDLL("artifacts/liberfinv_plugin.so")   # Python builder：parse 前加载
```
```bash
trtexec --onnx=customop_plugin.onnx --fp16 --staticPlugins=artifacts/liberfinv_plugin.so
```

结果是一个**完整的 TensorRT engine，零回退**，可被不带 Python 的 `trtexec --loadEngine` 直接运行。

**部署注意：**
- engine 只记录 plugin 名，不含代码，运行时必须加载 `.so`。
- 缺 `.so` 时在 `createExecutionContext` 才报错，**不在反序列化时** —— 只做
  `deserialize_cuda_engine` 的健康检查查不出来。
- `.so` 按 `-arch=sm_110a` 编译，换 GPU 架构要重编。

### 其他路线

- **dynamo_trt：** 零额外工作即可跑，但切成 3 段（2 个 linear 进 TRT，erfinv 留在 PyTorch），
  边界同步和显存往返使其比 eager 慢 3 倍。把同一 `.so` 注册为 Torch-TensorRT converter 可变为
  1 段 0 回退（0.118 ms，见第 10 章）。
- **ts_trt：** 回退段与 TRT engine 间 dtype 不协调（`Expected input tensors to have type Half, found type float`），
  构建正常、首次推理崩溃 —— 比构建失败更糟。
- **torch_compile：** Inductor 直接为 erfinv 生成 kernel，没有"不支持"的概念；比 eager 慢只因模型太小。

## 5.5 为什么 ONNX + plugin 看起来比 eager 慢

`exp_plugin_latency.py`（日志 `artifacts/plugin_latency.log`）拆解了这个问题。

**GPU 时间（nsys，1000 次平均）三者相当：**

| 路线 | kernel | GPU 时间 |
|---|---|---|
| eager | 2×GEMM + clamp + erfinv（4 个） | 9.4 µs |
| aoti | 2×GEMM + clamp/erfinv 融合（3 个） | 8.1 µs |
| trtexec + plugin | GEMM + GEMM/bias/clamp 融合 + plugin（3 个） | 9.8 µs |

32×512 的矩阵乘喂不饱 GPU，TensorRT 的优化在此量级省不出时间。端到端延迟（50–100 µs）
是 GPU 时间的 5–10 倍，**快慢完全由 host 开销决定**。

**同一个 `.plan`，换 runner（b=32，CUDA event median）：**

| 方案 | 延迟 | 加速比 |
|---|---|---|
| eager | 0.063 ms | 1.00x |
| aoti | 0.055 ms | 1.15x |
| TRT + plugin（教学用 `TrtRunner`） | 0.094–0.105 ms | 0.60–0.67x |
| TRT + plugin（精简封装 `LeanRunner`） | **0.034 ms** | **1.85x** |
| TRT + plugin（CUDA graph） | **0.022 ms** | **2.86x** |
| 参考：`trtexec --loadEngine`（C++） | 0.028 ms host / 0.021 ms GPU | — |

**慢的是 `TrtRunner`，不是 TensorRT。** 其开销构成：

| TrtRunner 每次调用的开销 | host 时间 |
|---|---|
| 基线：set shape/address + `execute_async_v3` | 0.019 ms |
| 每次 `torch.empty` 分配输出 + `get_tensor_shape` | +0.011 ms |
| 每次重建 dtype 映射 dict | +0.010 ms |
| **自建 stream + `stream.synchronize()`** | **+0.031 ms** |

最大项是自建 stream 加同步：host 无法与 GPU 重叠，GPU 时间被串行叠加。（另外 `bench_ms`
在当前 stream 打 event 而 kernel 在另一 stream 上，只是因为 TrtRunner 先同步了才测得"对"。）

`LeanRunner` 只做三件事：**预分配 buffer 并在构造时绑定地址、shape 不变时不重复设置、
在调用方 stream 上 enqueue 且不同步**。

**结论：**
- TRT 的优势来自"一次调用跑完整图、host 开销小"，**不是编译器优化**（GPU 时间相同）。
  小模型上比的是 host 开销（第 6 章规律 1）。
- 不能据此说 TRT 赢了 AOTI：两者 GPU 时间只差 1.7 µs，AOTI 的差距来自其 Python 包装器，
  未测 AOTI 的 C++ 用法；第 10 章 b=8192 计算受限时 AOTI 更快（0.144 vs 0.233 ms）。
- **零额外工作下，唯一比 eager 快且能脱离 Python 部署的是 AOTInductor**（`.pt2`）。
  ONNX + plugin + 合理封装也能做到，但要付出工程量。
- **写 plugin 前先考虑不写：** 导出期把 erfinv 分解为标准 ONNX 算子，无需 `.so`，
  且 TRT 能与相邻层融合（plugin 是融合屏障），第 10 章实测更快。

## 5.6 本章要记住的

1. **trace 类路线（ONNX 默认、TorchScript）在数据相关控制流上会静默出错**，还可能在延迟表上排第一。
   用 `torch.export` 当 linter 检查，用走不同分支的输入做测试。
2. **`torch.export` 的报错是特性。** 它拒绝的，trace 会照做但做错。
3. **不支持的算子对 ONNX 路线是工程量，不是死路：** 自定义 ONNX 算子 + TensorRT plugin
   （op_type = plugin 名）即可得到零回退的完整 engine；`dynamo_trt` 零成本能跑，但切段后慢 3 倍。
4. **评估时把子图数、GPU 时间、host 开销分开看。** 随手写的 runner 会让更快的 engine 看起来比 eager 还慢。

---

← [第 4 章：Torch-TensorRT 路线](04_torch_tensorrt_route.md) | [第 6 章：综合对比与选型](06_benchmark_comparison.md) →
