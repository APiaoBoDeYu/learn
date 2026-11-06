# 第 9 章：动态 shape —— 一个 engine 能不能替代 N 个

第 6 章列的第二个缺口。它重要是因为**静态 shape 在生产里常常是个谎**：
在线服务的 batch 随流量变，NLP 的序列长度每个请求都不同。绕过去的两个办法
（全部 pad 到最大长度、或者每个 shape 建一个 engine）都要花真钱，
所以各路线对动态性的支持程度是一个一等选型标准。

本章要回答的核心问题是：**一个动态 engine 能不能替代 N 个静态 engine？
代价是多少？**

## 9.1 三条路线，三种完全不同的表达方式

这个差异本身就是本章的主要内容。

### ONNX → TRT：两处配置，漏一处就静默变静态

```python
# 第一处：导出时告诉 ONNX 哪个维度是符号的
routes.build_onnx_trt(
    model, (bert_inputs(128),),
    dynamic_axes={"in0": {1: "seq"}, "out": {0: "batch"}},
    # 第二处：构建时告诉 TensorRT 支持什么范围、以及针对哪个 shape 调优
    profile=[("in0", (8, 32), (8, 128), (8, 384))],
)
```

**两处都必须配。** 只配 `dynamic_axes` → ONNX 图里有符号维度但 engine 是静态的；
只配 `profile` → 图里维度是固定数字，profile 无从生效。

而且 `(min, opt, max)` 这个三元组里，**TensorRT 只针对 `opt` 做 tactic 选型**，
min/max 只定义什么 shape 合法。这一点后面 9.4 会用数据说明它的后果有多大。

### torch.export：符号维度 + 显式边界

```python
seq = torch.export.Dim("seq", min=32, max=384)
routes.build_dynamo_trt(model, (bert_inputs(128),),
                        dynamic_shapes={"ids": {1: seq}})
```

**一处配置**，Torch-TensorRT 自动把 `Dim` 的边界翻译成 TRT profile。
而且 `Dim` 在导出期是**符号检查**的：如果你的模型里有某个操作只对特定长度成立，
`torch.export` 会在导出时就报错，而不是等到推理时给你错结果。
这是第 5 章那条"严格性是特性"在动态 shape 上的延续。

### torch.compile：什么都不用声明

```python
compiled = torch.compile(model, dynamic=True)
```

没有边界、没有 profile。Dynamo 在遇到新 shape 时检查 guard，
不满足就重新编译。代价是**首次遇到新 shape 要付编译时间**，
好处是零配置、无边界限制。

## 9.2 实测结果

`bert_tiny`，batch=8，FP16，序列长度 32 / 128 / 384。
所有动态 engine 的 `opt` 点都设在 **seq=128**。

```
### latency by sequence length (ms)

| route | seq=32 | seq=128 | seq=384 |
|---|---|---|---|
| eager (fp16) | 0.809 | 0.797 | 0.807 |
| onnx_trt dynamic | 0.401 | 0.457 | 0.742 |
| dynamo_trt dynamic | **0.285** | **0.359** | **0.698** |
| torch_compile dynamic | 0.731 | 0.728 | 0.731 |
| onnx_trt static (每个长度一个 engine) | **0.218** | **0.286** | **0.515** |
```

所有路线在所有长度上 `cos = 1.00000` —— 动态 shape **不引入数值问题**，
这一章的所有代价都是性能代价。

### 发现 1：动态 shape 下 dynamo_trt 明显优于 ONNX 路线

这**推翻了第 6 章的"推翻 1"**。第 6 章在静态 shape 下测出两条路线差异 <5%，
结论是"迁移到 Torch-TensorRT 换不来性能"。动态 shape 下完全不是这样：

| seq | onnx_trt | dynamo_trt | dynamo 优势 |
|---|---|---|---|
| 32 | 0.401 | 0.285 | **+29%** |
| 128 | 0.457 | 0.359 | **+21%** |
| 384 | 0.742 | 0.698 | +6% |

**在最常见的短序列上快 20–29%**，这远超测量噪声。

原因大概是两条路线生成的 shape 处理代码不同：`torch.export` 的符号 shape 信息
能更完整地传给 TensorRT，而 ONNX 路线经过一次 IR 转换后，有些 shape 约束退化成了
运行时计算。第 3 章说的"少一次 IR 转换就少一类问题"，在动态 shape 上第一次体现为
**性能差异**而不只是正确性差异。

所以第 6 章那条建议需要修正：

> **"已有 ONNX 路线跑得好就别动"只适用于静态 shape。
> 如果你在跑动态 shape，迁移到 Torch-TensorRT 可能有 20% 以上的收益，值得实测。**

### 发现 2：torch.compile 的延迟对 shape 完全不敏感

0.731 / 0.728 / 0.731 —— **三个长度几乎一模一样**，尽管序列长度差了 12 倍。

这不是好事。它说明这个尺寸下 `torch.compile` 的延迟被**固定开销**
（guard 检查、Python 调度、kernel 启动）主导，实际计算量的变化淹没在里面。
对照 eager 也是平的（0.809/0.797/0.807），而两条 TRT 路线都随长度明显增长 ——
说明只有 TRT 路线真正在按计算量花时间。

实用含义：`torch.compile(dynamic=True)` 在短序列上（0.731 vs dynamo_trt 的 0.285）
劣势很大，但在长序列上差距收窄（0.731 vs 0.698，几乎追平）。
**序列越长、计算越密，torch.compile 越有竞争力。**

## 9.3 动态性的真实价格

把动态 engine 和**专门为该长度构建的静态 engine**比：

```
| seq | dynamic onnx_trt | static onnx_trt | 动态的代价 |
|---|---|---|---|
| 32 | 0.401 | 0.218 | **+84.1%** |
| 128 | 0.457 | 0.286 | **+59.7%** |
| 384 | 0.742 | 0.515 | **+44.2%** |
```

**这是本章最重要的数字。动态 engine 比专用静态 engine 慢 44–84%。**

注意即使在 `opt` 点（seq=128，engine 明确为它调优过）也慢 **59.7%**。
这一点值得强调，因为它违反直觉：很多人以为 `opt` 点上动态 engine 应该和静态
engine 差不多。实测不是 —— 支持一个 shape 范围本身就要求 TensorRT 选择
**更通用因而更慢**的 kernel，并插入运行时 shape 计算，这个成本在整个范围内都要付。

而且离 `opt` 越远代价越大：seq=32（离 opt 最远，且绝对计算量最小，
固定开销占比最高）代价 84%。

### 这个数字怎么用

它把选型问题变成一道算术题：

- **shape 种类少（2–3 种）且可枚举** → **建多个静态 engine**，用一个 dict 分发。
  多花一点显存和构建时间，换 44–84% 的性能。这几乎总是对的选择。
- **shape 连续变化、无法枚举** → 动态 engine，接受这个代价。
- **中间情况** → 分桶（bucketing）：建几个静态 engine 覆盖常见区间，
  把输入 pad 到最近的桶。这是生产上最常见的做法，因为 pad 的浪费通常
  远小于 44–84%。

举个具体的算术：如果你的请求 80% 落在 seq≤128，pad 到 128 的浪费大概是
把平均 64 长度的请求算成 128，浪费约 2 倍计算 —— 但静态 engine 快 60%，
而且短序列本来就便宜。**这道题经常算下来是静态 engine 赢。**

## 9.4 关于 `opt` 点的一个警告

TensorRT 只针对 `opt` 做 tactic 选型。本章的数据能反推出这个选择有多重要：
动态 engine 在 seq=32 上的代价（84%）远大于 seq=384（44%），
而 `opt` 设在 128。

**推论：`opt` 要设成你线上最常见的 shape，不是 min 和 max 的中间值。**
如果你线上 90% 的请求是 seq=32，却把 `opt` 设成 208（32 和 384 的中点），
你会为那 10% 的长请求优化，而让 90% 的请求跑在次优 kernel 上。

第 3 章提过这一点，本章给了它一个量化的量级。

## 9.5 踩到的一个真 bug：`offload_module_to_cpu` 会污染后续所有路线

这个实验第一次跑的时候，**三条路线全部失败**：

```
--- dynamo_trt dynamic ---
  seq=  32: FAILED RuntimeError: Expected all tensors to be on the same device,
            but got index is on cuda:0, different from other tensors on cpu
--- torch_compile dynamic ---
  BUILD FAILED: TorchRuntimeError: Dynamo failed to run FX node with fake tensors
[static onnx_trt seq=32] FAILED RouteUnsupported
```

三个完全不同的报错，看起来像三个独立的 bug。实际是**一个**根因。

排查过程（值得记录，因为方法比结论更有用）：先单独跑静态 engine 构建 —— 成功。
再按实验里的顺序跑（先 dynamo_trt，再静态）—— 失败。于是问题不在任何单条路线，
而在**它们之间的相互影响**。最小复现：

```python
model, _ = models.get("bert_tiny", seq=128)
print("before:", next(model.parameters()).device)   # cuda:0
routes.build_dynamo_trt(model, ...)
print("after :", next(model.parameters()).device)   # cpu   ← 元凶
```

`offload_module_to_cpu=True`（第 1 章为了绕开 Jetson 统一内存的显存告警而加的）
**会把调用方的模型原地搬到 CPU，并且不搬回来**。之后任何对这个 model 的使用
—— 下一条路线、准确率参考、ONNX 导出 —— 都会失败。

修法是在 `routes.py` 里记下原始 device 并在 `finally` 里恢复：

```python
_orig_devices = {n: p.device for n, p in
                 list(model.named_parameters()) + list(model.named_buffers())}
...
finally:
    _restore()
```

修完之后三条路线全部正常。

**两个教训**：

1. **编译 API 可能原地修改你的模型。** 这类副作用不会写在显眼处，
   而且症状出现在**完全无关的后续代码**里。做多路线对比时，
   每条路线之后检查一下模型状态是值得的。
2. **"三个不同报错"不代表"三个 bug"。** 如果多条独立路线同时开始失败，
   先怀疑它们共享的状态，而不是逐个去调。

顺带说，这个 bug 影响的不只是本章 —— 它一直潜伏在前面几章的代码里，
只是之前的实验恰好每条路线都重新构建模型，所以没暴露。

## 9.6 各路线动态 shape 支持总结

| 路线 | 怎么声明 | 配置点数 | 边界检查 | 动态实测表现 |
|---|---|---|---|---|
| onnx_trt | `dynamic_axes` + `profile` | **2** | 无（漏配静默变静态） | 0.401–0.742 |
| dynamo_trt | `torch.export.Dim` | **1** | **有**（导出期符号检查） | **0.285–0.698（最快）** |
| torch_compile | `dynamic=True` | **0** | 无（运行时重编译） | 0.731（对 shape 不敏感） |
| 静态多 engine | 每个 shape 一次构建 | N | N/A | **0.218–0.515（最快，但不灵活）** |

## 9.7 动手

```bash
cd /root/data/learn/torch2trt/code
python3 exp_dynamic_shape.py        # 约 10 分钟
```

建议自己试的三件事：

1. **改 `OPT_SEQ`**（比如设成 32），重跑，看动态 engine 的代价分布怎么移动。
   这是理解 `opt` 点重要性最直接的方法。
2. **把 batch 也设成动态**（profile 里 batch 维给范围，`Dim` 加一个 batch 维），
   看两个动态维度的代价是否叠加。
3. **在 `torch_compile` 路线上开 `TORCH_LOGS=recompiles`**，数它到底重编译了几次 ——
   验证"零配置"的隐藏成本。

---

← [第 8 章：量化](08_quantization.md) | [第 10 章：自定义算子与 plugin](10_custom_op_and_plugin.md) →
