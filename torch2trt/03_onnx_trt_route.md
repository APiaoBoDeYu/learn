# 第 3 章：R2 / R3 —— torch → ONNX → TensorRT

这是你现在在走的路线。本章的目标不是教你怎么走（你已经会了），而是：

1. 把这条路线的**每一步在做什么、坑在哪**说清楚；
2. 指出它和后面几条路线在**架构上**的根本差异（不是性能差异）；
3. 用实测数据说明 R2（Python API）和 R3（trtexec CLI）该怎么选。

## 3.1 两步走，两类坑

```
nn.Module ──[torch.onnx.export]──> ONNX 文件 ──[TRT OnnxParser+Builder]──> .plan
             ↑ 坑类型 A                          ↑ 坑类型 B
             语义/opset 问题                      算子支持/精度/profile 问题
```

**这条路线最大的工程特点：两步分别属于两个不同的项目，报错信息也分属两个世界。**
PyTorch 那边说"opset 17 不支持 aten::erfinv"，TensorRT 那边说
"No importer registered for op: Erfinv"。同一个根因，两种说法，在两个阶段暴露。

这既是缺点也是优点：缺点是排查路径长；优点是**中间那个 ONNX 文件是可检查的产物**。
你可以用 Netron 打开它看图结构，可以用 `onnx.checker` 验证，可以用 polygraphy 单独
跑它 —— 后面几条路线没有这个可观测的中间态。这是 R2/R3 被长期偏爱的真实原因之一。

## 3.2 第一步：ONNX 导出

`routes.py` 里的实现：

```python
def export_onnx(model, inputs, path, dynamic_axes=None, opset=17, dynamo=False):
    try:
        _, secs = _timed(lambda: torch.onnx.export(
            model, inputs, str(path),
            opset_version=opset,
            input_names=[f"in{i}" for i in range(len(inputs))],
            output_names=["out"],
            dynamic_axes=dynamic_axes,
            dynamo=dynamo,
        ))
        return secs
    except Exception as e:
        raise RouteUnsupported(f"onnx export failed: {e}") from e
```

四个参数值得单独说：

**`opset_version`**：ONNX 算子集版本。**这是 PyTorch 和 TensorRT 之间的握手协议**，
选错就白费。规律是：opset 太低 → PyTorch 导不出新算子；opset 太高 → TensorRT 的
parser 不认。TensorRT 10.x 稳妥选 17。这个"两头夹逼"的版本约束是 ONNX 路线独有的
成本，torch.export 路线没有（它没有第三方算子集）。

**`input_names` / `output_names`**：**别省**。不指定的话 ONNX 会生成
`input.1`、`onnx::Gemm_0` 这类自动名字，而你后面设 optimization profile
（`profile.set_shape(name, ...)`）必须用名字索引。自动名字会在模型结构微调后变化，
于是你的部署脚本莫名失效。

**`dynamic_axes`**：标记哪些维度是动态的。**不写就是静态的**，engine 只接受导出时
那个确切 shape。这是新手最常见的困惑来源：模型明明能跑 batch=1 也能跑 batch=8，
导出的 engine 却只吃 batch=8。

**`dynamo=True/False`**：PyTorch 2.x 新增的选项，走 `torch.export` 抓图再转 ONNX，
而不是走老的 trace。它修掉了一部分 trace 的语义丢失问题，但也继承了 `torch.export`
的严格性（第 5 章会看到：严格性有时是好事）。本教程默认 `False`（传统 trace 路径），
因为那才是绝大多数现有生产代码在走的。

## 3.3 第二步：TensorRT 构建（R2，Python API）

```python
builder = trt.Builder(logger)
network = builder.create_network()
parser = trt.OnnxParser(network, logger)
with open(onnx_path, "rb") as f:
    if not parser.parse(f.read()):
        errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RouteUnsupported("onnx parse failed: " + "; ".join(errs))
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
if fp16:
    config.set_flag(trt.BuilderFlag.FP16)
if profile:
    p = builder.create_optimization_profile()
    for iname, mn, opt, mx in profile:
        p.set_shape(iname, mn, opt, mx)
    config.add_optimization_profile(p)
plan = builder.build_serialized_network(network, config)
```

关键点：

**`parser.parse()` 返回 bool，不抛异常。** 不检查返回值会得到一个空 network，
然后在后面某处以一个完全无关的错误崩掉。错误信息必须自己从 `get_error(i)` 循环取。
这是 TensorRT Python API 的 C++ 出身痕迹，也是本教程把它包成 `RouteUnsupported`
的原因。

**`FP16` 是一个 flag，不是承诺。** 设了它只是**允许** TensorRT 用 FP16 tactic；
builder 仍会按它的性能模型为每层挑选，某些层可能留在 FP32。想强制可以加
`BuilderFlag.OBEY_PRECISION_CONSTRAINTS`，但通常会更慢 —— builder 的选择一般是对的。

**workspace 是上限，不是预分配。** 给大了不会浪费显存，只是允许 builder 尝试更耗
内存的 tactic。给太小会让它错过最快的实现。

**optimization profile 是动态 shape 的全部关键。** `(min, opt, max)` 三元组里，
TensorRT 只针对 `opt` 那个 shape 做 tactic 选型。所以 `opt` 要设成你**线上最常见**
的 shape，不是中间值。设成 `(1, 1, 128)` 而线上常跑 64，你会得到一个为 batch=1
优化、跑 batch=64 时很慢的 engine。

## 3.4 第三步：你得自己写运行时

这是 ONNX→TRT 路线一个容易被低估的成本。builder 给你的是一段 `bytes`，
**怎么跑它是你的事**。`routes.py` 里的 `TrtRunner` 是最小实现：

```python
class TrtRunner:
    def __init__(self, plan, logger):
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(plan)
        self.ctx = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        ...

    def __call__(self, *args):
        for name, tensor in zip(self.in_names, args):
            tensor = tensor.contiguous()
            want = self._torch_dtype(name)
            if tensor.dtype != want:
                tensor = tensor.to(want)      # int64 ids -> int32
            self.ctx.set_input_shape(name, tuple(tensor.shape))
            self.ctx.set_tensor_address(name, tensor.data_ptr())
        for name in self.out_names:
            shape = tuple(self.ctx.get_tensor_shape(name))
            buf = torch.empty(shape, dtype=self._torch_dtype(name), device="cuda")
            self.ctx.set_tensor_address(name, buf.data_ptr())
        self.ctx.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
```

这约 40 行是你选这条路线时**签下的维护责任**：绑定管理、device buffer 分配、
stream、dtype 转换、动态 shape 设置。都不难，但都是你的代码、你的 bug。

对比 R4（dynamo_trt）：那条路线返回的东西**已经是一个 torch Module**，
直接 `model(x)` 就行，这 40 行不存在。这是架构差异，不是性能差异，
但对小团队常常更重要。

### `dtype` 那一段不是多余的

`bert_tiny` 的输入是 int64 token ids，但 TensorRT 长期以来对 INT64 支持有限，
ONNX 导出 + TRT 构建的组合经常把它降成 INT32。于是你 `set_input_shape` 时
engine 期待 int32，而 torch 手里是 int64。

```python
want = self._torch_dtype(name)
if tensor.dtype != want:
    tensor = tensor.to(want)
```

真实服务代码最后都会长出这一段。注意这里有个实现细节：cast 产生的是**新张量**，
必须在 `execute_async_v3` 之前保持引用存活，否则 Python GC 会把那块显存回收，
engine 读到垃圾。`TrtRunner` 里那句 `self._keep = tensor` 就是干这个的 ——
一个只在"手写运行时"路线上才需要你操心的问题。

## 3.5 R3：同一个 ONNX，改用 trtexec

```bash
trtexec --onnx=resnet18.onnx --saveEngine=resnet18.plan --fp16
```

`routes.py` 里就是把它当子进程调：

```python
cmd = ["trtexec", f"--onnx={onnx_path}", f"--saveEngine={engine_path}"]
if fp16:
    cmd.append("--fp16")
proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
if proc.returncode != 0:
    raise RouteUnsupported(f"trtexec failed: ...")
```

**产出的 engine 和 R2 是同一类东西**，所以本教程用同一个 `TrtRunner` 加载它 ——
这也顺便证明了这一点。

那为什么还要单列一条路线？因为**工作流不同**：

| | R2 Python API | R3 trtexec CLI |
|---|---|---|
| 构建环境 | 需要 Python + tensorrt 绑定 | 只需一个二进制 |
| 适合放在 | 训练脚本、实验 notebook | Dockerfile、CI step、Makefile |
| 精细控制 | 全部 builder API 可用 | 受限于命令行 flag |
| 自带 profiling | 没有，自己写 | **有**，`--dumpProfile` 给逐层耗时 |
| 调试动态 shape | 自己拼 profile 代码 | `--minShapes/--optShapes/--maxShapes` |

实践中最好的用法是**两条都用**：拿 trtexec 做快速探查（它的 `--dumpProfile` 和
`--verbose` 是排查"为什么这个 engine 慢"最快的工具），确定配置后再用 Python API
固化到构建流程里。

一个非常实用的 trtexec 用法，不需要你写任何代码就能拿到逐层耗时：

```bash
trtexec --onnx=model.onnx --fp16 --dumpProfile --separateProfileRun
```

## 3.6 实测：R2 vs R3

resnet18（batch=8，FP16），本机实测：

| | 构建耗时 | 中位延迟 | engine 大小 |
|---|---|---|---|
| R2 onnx_trt | 17.8 s | 0.852 ms | 23.6 MB |
| R3 trtexec | 21.6 s | 0.826 ms | 23.6 MB |

ctrlflow（batch=32）：

| | 构建耗时 | 中位延迟 |
|---|---|---|
| R2 onnx_trt | 5.3 s | 0.104 ms |
| R3 trtexec | 10.4 s | 0.103 ms |

**结论：延迟基本一致**（差异在测量噪声量级内），符合预期 —— 同一个 builder、
同一个 ONNX、同一套 tactic 选型。R3 构建稍慢的原因是进程启动 + engine 落盘再读回。

所以 R2 vs R3 **不是性能选择，是工作流选择**。别指望换 CLI 能变快。

## 3.7 这条路线的真实优缺点

先说优点，因为它确实是被广泛采用的默认选择，理由是充分的：

- **产物脱离 Python**。`.plan` 文件可以被 C++ / Triton Inference Server / DeepStream
  直接加载。这常常是硬需求，和性能无关。
- **中间态可观测**。ONNX 文件能用 Netron 看图、用 polygraphy 单独跑、用
  onnx-graphsurgeon 手术式修改。这是排查"为什么转换后结果不对"最有力的抓手，
  也是别的路线缺的。
- **生态最成熟**。踩到的坑基本都有人踩过并写过帖子。
- **跨框架**。同一套 TRT 构建代码能吃 TensorFlow/JAX 导出的 ONNX。

缺点，按实际影响排序：

- **默认全有或全无**。一个算子不支持，整条路线断掉。有两个逃生口，正好对应两步：
  导出期写 symbolic 生成**自定义 ONNX 算子**（或分解成标准算子），构建期给 TRT 一个
  **plugin**（C++ `IPluginV3` 或 Python `tensorrt.plugin`）。第 10 章实测了全部组合 ——
  结论是最好的那个往往不用写 plugin。第 5 章会实测 R4 在这件事上的另一种做法。
- **两次 IR 转换，两类报错。** opset 是一个额外的版本约束维度。
- **trace 语义丢失**。默认的 `torch.onnx.export` 走 trace，**数据相关控制流会被
  静默固化**。第 5 章会证明这是本教程测到的最危险的行为。
- **运行时代码是你的**。那 40 行 `TrtRunner`。
- **动态 shape 要手工配 profile**，而且 `opt` 选错会安静地损失性能。

## 3.8 动手

```bash
cd /root/data/learn/torch2trt/code
python3 run_bench.py --models resnet18,bert_tiny --routes onnx_trt,trtexec
ls -la ../artifacts/*.onnx ../artifacts/*.plan
```

可以用 `trtexec --loadEngine=../artifacts/resnet18_cli.plan --dumpProfile` 看逐层耗时，
观察 TensorRT 把哪些层融合掉了（你会看到 conv+bn+relu 变成一个 `...+Relu` 条目，
而 ONNX 图里它们是三个独立节点）。

---

← [第 2 章：基准与模型设计](02_baseline_and_models.md) | [第 4 章：Torch-TensorRT 路线](04_torch_tensorrt_route.md) →
