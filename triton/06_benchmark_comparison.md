# 第 6 章：综合实战对比

## 6.1 测试对象

复用第 4 章的 MLP block（Linear → GELU → Linear → 残差 → LayerNorm，`dim=1024, hidden=4096,
batch=256`），在同一个模型上跑五条路径，全部代码见
[`code/06_benchmark_all.py`](code/06_benchmark_all.py)：

1. **Eager**：纯 PyTorch，什么都不做，作为基线
2. **TorchScript**（`torch.jit.script`）：你已经熟悉的路径
3. **torch.compile**（`default` 和 `max-autotune` 两种 mode）
4. **手写 Triton kernel**：LayerNorm 部分换成第 3 章写的 fused kernel，其余保持 eager
5. **ONNX → TensorRT**：`torch.onnx.export` 导出，`trtexec` 构建 FP16 engine 并测速

## 6.2 关于 ONNX 导出的一个真实小插曲

写这一章时，默认调用 `torch.onnx.export(...)` 直接报错：

```
ModuleNotFoundError: No module named 'onnxscript'
```

原因：PyTorch 从某个版本起把**默认导出器**换成了基于 `torch.export` 的新导出器（`dynamo=True`），
这个新导出器依赖 `onnxscript` 包，本机没装。而老的、基于 TorchScript trace 的导出器
(`dynamo=False`) 不需要这个依赖，功能上对这个简单模型完全够用：

```python
torch.onnx.export(
    model, (x,), onnx_path,
    input_names=["x"], output_names=["out"],
    dynamic_axes={"x": {0: "batch"}, "out": {0: "batch"}},
    opset_version=17,
    dynamo=False,  # 用老的 TorchScript-based 导出器，避免额外装 onnxscript
)
```

选择 `dynamo=False` 而不是装 `onnxscript`，是因为本教程的原则是"不为了凑对比引入不必要的
新依赖"——环境里已有的 `tensorrt`、`onnx`、`trtexec` 已经足够完成这条路径的验证。
（`onnxruntime` 本机也没装，本章的 TensorRT 路径直接用 `trtexec` 命令行构建/跑 engine，
没有用 `onnxruntime` 做中间验证，同样是为了不引入非必要依赖。）

## 6.3 实际运行结果

```bash
python3 code/06_benchmark_all.py
```

正确性校验全部通过：

```
[PASS] torchscript: allclose=True, max_abs_err=0.000e+00
[PASS] torch.compile default: allclose=True, max_abs_err=9.835e-07
[PASS] torch.compile max-autotune: allclose=True, max_abs_err=9.835e-07
[PASS] hand-written Triton LN: allclose=True, max_abs_err=9.537e-07
```

延迟汇总（batch=256, dim=1024, 本机 NVIDIA Thor 实测）：

| 路径 | 延迟 (ms) | 相对 Eager 加速 |
|---|---|---|
| eager | 0.2884 | 1.00x |
| torchscript | 0.2869 | 1.01x |
| compile (default) | 0.2587 | 1.12x |
| compile (max-autotune) | 0.2691 | 1.07x |
| 手写 Triton LN + 其余 eager | 0.2934 | 0.98x |
| **ONNX → TensorRT (FP16)** | **0.0596** | **4.84x** |

## 6.4 如何解读这组数字

**TensorRT 遥遥领先**——这是符合预期的真实结果，原因和"AOT vs JIT"直接相关：

- TensorRT 在**离线 build 阶段**拿到完整的计算图，可以做跨层融合、精度转换（这里用了
  FP16，直接减半了内存带宽需求和部分计算量）、为这个具体 shape 挑选最优 kernel/tactic——
  这些决策不受"要在几百毫秒内完成一次 JIT 编译"的时间预算限制，可以搜索得更充分
- torch.compile 是 JIT，为了保证首次调用的编译时间可接受（用户不会接受推理请求等好几秒
  编译完成），Inductor 的融合/搜索策略天然更保守
- 手写 Triton LN 版本只融合了一个算子（LayerNorm），其余部分（两次 Linear、GELU）仍是
  未融合的 eager 调用，融合覆盖面比 torch.compile 的全图优化小，所以反而和纯 eager
  基本持平（0.98x，在测量噪声范围内）
- TorchScript 在这个模型上几乎没有加速（1.01x）——`torch.jit.script` 本身**不做**
  算子融合优化，它只是把 Python 换成了一份可以脱离 Python 解释器执行的 IR，性能提升
  主要来自减少 Python 解释开销，对这种已经是大算子（大矩阵乘）主导的模型收益很小

**这组结果传递的核心信息**：AOT（TensorRT）和 JIT（torch.compile）不是"谁更好"的关系，
是**不同部署场景下的合理选择**，第 6.5 节展开讲怎么选。

## 6.5 选型指南

**选 TensorRT，当：**
- 目标是生产部署，尤其是需要脱离 Python（C++ 服务、边缘设备固件）、跨语言调用
- 输入 shape 范围提前可知，能接受为 shape 范围离线 build engine 的部署流程
- 追求单次推理延迟的极限优化，能接受 AOT 编译的额外工程成本（导出 ONNX、调试转换问题、
  维护 engine 版本与硬件/驱动版本的绑定关系）
- 本章实测：同一模型 TensorRT 比 eager 快 4.84 倍，是五条路径里最快的

**选 torch.compile，当：**
- 还在 Python/PyTorch 生态里做训练或推理，想要"几乎不改代码"拿到一定加速
- 模型结构会变动、迭代速度优先于极限性能（JIT 意味着改代码后直接重跑就行，不需要重新
  走一遍导出/build 流程）
- 需要处理动态 shape、依赖 Python 控制流的模型（第 5 章的 graph break 机制保证正确性）
- 本章实测：1.07x～1.13x，加速有限但零额外部署链路复杂度

**何时应该手写 Triton kernel：**
- torch.compile/Inductor 自动生成的融合方案性能不够，且你能设计出比自动生成更优的融合
  边界（本教程第 3 章的 fused LayerNorm 独立测试时确实比多 kernel 快 5.2 倍，但接入
  一个更大模型、其余部分不融合时优势会被稀释——这也是本章手写 Triton 版本没有展现优势
  的原因：**必须和 torch.compile 结合，让自定义 kernel 参与全图融合，才能同时拿到
  "融合正确"和"覆盖全图"两个收益**，见第 5.3 节的 `triton_op` 接入方式）
- 需要实现 PyTorch/Inductor 没有原生支持的算子或融合模式（如自定义 attention 变种）
- 需要精确控制某个 kernel 的资源使用（shared memory、寄存器分配），比 Inductor 自动决策
  更贴近你对硬件的理解

**TorchScript 现在的定位**：如果你在新项目里做选择，`torch.compile`（训练/迭代阶段）+
TensorRT 或 AOTInductor（生产部署阶段，见第 7 章延伸阅读）基本覆盖了 TorchScript 原来的
使用场景，且都比 TorchScript 更好用（兼容性更好、有实质性能收益）。TorchScript 更多是
维护旧代码库时才会继续接触。

下一章：[第 7 章：附录](07_appendix.md) —— 调试技巧汇总、环境变量速查、常见坑
