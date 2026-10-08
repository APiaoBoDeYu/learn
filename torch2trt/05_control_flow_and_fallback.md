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

## 5.5 第二个实验：不支持的算子

`exp_unsupported_op.py`，模型含 `torch.erfinv`（正常 aten 算子，但无 ONNX opset
映射、无 TRT converter）：

```
| route | outcome | median ms | TRT subgraphs | fallback subgraphs |
|---|---|---|---|---|
| eager | built | 0.066 | - | - |
| torch_compile | built | 0.144 | - | - |
| onnx_trt | build_failed | - | - | - |
| dynamo_trt | built | 0.192 | 2 | 1 |
| ts_trt | build_failed | - | - | - |
```

四种行为：

**`onnx_trt` —— 第一步就死。**

```
UnsupportedOperatorError: Exporting the operator 'aten::erfinv' to ONNX
opset version 17 is not supported
```

值得注意的是**它死在 PyTorch 侧，不是 TensorRT 侧**。连 ONNX 文件都没生成。
所以这类问题你未必能怪 TensorRT —— 上一章说的"两步两类坑"，这是第一类。

好消息是报错清楚，还打出了 Torch IR 图，`aten::erfinv` 那一行一眼可见。

> 本章把这一行记成"构建失败"，是**不做任何额外工作**时的结果。ONNX 路线还有两个
> 逃生口（自定义 ONNX 算子 + TensorRT plugin，或分解成标准算子），第 10 章单独实测。

**`dynamo_trt` —— 切 3 段，能跑。** `TRT subgraphs=2, fallback=1`：
两个 `linear` 各自进了 engine，`erfinv` 留在 PyTorch。数值正确
（`cos_sim=0.999998, top1_agree=1.000`）。

**但 0.192ms vs eager 的 0.066ms —— 慢了 2.9 倍。** 上一章解释过：
3 个算子的模型切 3 段，边界的同步和显存往返完全压过了计算本身。

**`ts_trt` —— 构建/运行时崩溃。** 部分回退时 dtype 和 device 没协调好
（详见上一章）。

**`torch_compile` —— 毫无问题。** Inductor 直接为 `erfinv` 生成 kernel，
没有"不支持"这个概念。它慢于 eager（0.144 vs 0.066）纯粹是因为模型太小，
编译收益盖不住开销。

### 结论：`customop` 上正确答案是 AOTInductor

把完整数据放一起：

| 路线 | customop 延迟 | 加速比 | 能部署脱离 Python 吗 |
|---|---|---|---|
| eager | 0.068 ms | 1.00x | 否 |
| **aoti** | **0.060 ms** | **1.14x** | **是** |
| torch_compile | 0.135 ms | 0.50x | 否 |
| dynamo_trt | 0.192 ms | 0.34x | 否 |
| onnx_trt / trtexec / ts_trt | 构建失败 | — | — |

**在这个模型上，不做额外工作的前提下，唯一既比 eager 快、又能产出脱离 Python 的
产物的路线是 AOTInductor —— 一条完全不碰 TensorRT 的路线。**

第 10 章补上了 ONNX 路线的自定义算子 + plugin 方案：它们能让 ONNX 路线构建成功并
脱离 Python，最好的也只到 0.079 ms（Python 调用侧），**AOTI 仍是最快的**。这条结论成立。

这是整个教程里最反直觉的一条结论，也是做这些实验的意义：如果只按"TensorRT 是
NVIDIA 推理的标准答案"来选，在这类模型上你会选到一个比什么都不做还慢 3 倍的方案。

## 5.6 本章要记住的三件事

1. **trace 类路线（ONNX 默认、TorchScript）在数据相关控制流上会静默出错**，
   而且可能在延迟表上排第一。用 `torch.export` 当 linter 去查。
2. **`torch.export` 的报错是特性不是缺陷。** 它拒绝的东西，trace 会照做但做错。
3. **"能转"不等于"该转"。** `dynamo_trt` 让 `customop` 从"构建失败"变成"能跑"，
   但也比 eager 慢 3 倍。部分回退的子图计数必须看，它直接决定值不值得。
   ONNX 路线的自定义算子 / plugin 方案也一样，见第 10 章。

---

← [第 4 章：Torch-TensorRT 路线](04_torch_tensorrt_route.md) | [第 6 章：综合对比与选型](06_benchmark_comparison.md) →
