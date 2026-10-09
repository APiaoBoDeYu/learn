# 第 10 章：ONNX 自定义算子 + TensorRT plugin —— 前面几章跳过的两个逃生口

第 5 章把 `customop / onnx_trt` 记成"构建失败"就结束了，第 3、4 章也一直说
ONNX→TRT 是"全有或全无"。**这个说法不公平。** 它只描述了"遇到第一个报错就停下"
的情况。在生产里，ONNX 路线遇到不支持的算子时有两个标准做法，正好对应它的两步：

```
nn.Module ──[torch.onnx.export]──> ONNX ──[TRT OnnxParser + Builder]──> .plan
             ↑ 逃生口 1                      ↑ 逃生口 2
             自定义 ONNX 算子                  TensorRT plugin
             (symbolic / translation table)   (C++ IPluginV3 或 Python QDP)
```

- **逃生口 1（eager → ONNX）**：告诉导出器"遇到 `aten::erfinv` 时生成什么节点"。
  可以生成一个**自定义域节点**（`custom::ErfInv`），也可以**分解成一串标准 ONNX 算子**。
- **逃生口 2（ONNX → engine）**：给 TensorRT parser 一个能实现那个节点的 **plugin**。
  parser 遇到不认识的 op_type 时，会按名字去 plugin registry 里找 creator。

本章把这两个逃生口的所有实际可行组合都在同一个 `customop` 模型上跑了一遍，
和第 5 章的 `dynamo_trt` 部分回退、AOTInductor 放在一起比。

复现：

```bash
cd /root/data/learn/torch2trt/code
bash plugins/build.sh                  # 编译 C++ plugin → artifacts/liberfinv_plugin.so
python3 exp_custom_plugin.py           # b=32（与第 5/6 章一致）
python3 exp_custom_plugin.py --batch 8192   # 计算量大的情形
```

## 10.1 结论先放前面

1. **"全有或全无"不成立。** 加上 C++ plugin 后 ONNX 路线在 `customop` 上构建成功、
   数值正确（cos 0.999999）、**engine 能被不带 Python 的 trtexec 直接加载**。
   第 5 章"唯一能脱离 Python 的是 AOTI"这句话要改。
2. **但 AOTInductor 仍是这个模型上最快的。** b=32：AOTI 0.060 ms，ONNX+C++ plugin
   0.079 ms；b=8192：AOTI 0.144 ms，ONNX+plugin 0.233 ms。plugin 修好的是
   "能不能"，不是"快不快"。
3. **最好的 ONNX 方案常常根本不需要 plugin。** 在导出期把 `erfinv` 分解成 73 个标准
   ONNX 算子，TensorRT 会把它们和前面的 `clamp` 融成**一个** kernel ——
   b=8192 时纯 GPU 时间 0.159 ms，比 C++ plugin 的 0.179 ms 快 11%。
   **plugin 是一道融合屏障**，手写的东西 TensorRT 不会和邻居合并。
4. **Python plugin（QDP）在这个版本上是性能陷阱。** JIT 版运行时要 Python，
   AOT 版每次 enqueue 都重新 `cuModuleLoadData` 一次 PTX（nsys 实测 200 次推理
   202 次加载），一个 erfinv 要 0.16–0.18 ms，是 C++ plugin 的 **40 倍**。
5. **plugin 同样能救 Torch-TensorRT。** 把同一个 `.so` 注册成 converter，
   `dynamo_trt` 从"2 个 TRT + 1 个回退"变成"1 个 TRT + 0 回退"，
   延迟从 0.201 降到 0.118 ms（b=32）。

## 10.2 逃生口 1：让 eager → ONNX 能过

### 1a. 传统导出器（`dynamo=False`）：`register_custom_op_symbolic`

```python
def erfinv_symbolic(g, x):
    return g.op("custom::ErfInv", x).setType(x.type())

torch.onnx.register_custom_op_symbolic("aten::erfinv", erfinv_symbolic, 17)
torch.onnx.export(model, inputs, path, opset_version=17,
                  custom_opsets={"custom": 1}, dynamo=False)
```

symbolic 函数拿到 ONNX 图构造器 `g`，返回什么节点就生成什么。导出的图：

```
Gemm → Clip → custom::ErfInv → Gemm
```

几个细节：
- **`custom_opsets` 要声明域版本**，否则 `onnx.checker` 不认。
- **`.setType(x.type())` 别省**。自定义节点 ONNX 不知道输出类型，不写的话
  后续节点的 shape 推断会断。
- 节点名（`ErfInv`）就是 TensorRT 后面要找的 plugin 名，**两边必须完全一致**。

### 1b. 新导出器（`dynamo=True`）：`custom_translation_table`

```python
import onnx.defs as D, onnxscript

# 坑 1：onnxscript 按 OpSchema 解析 custom.ErfInv，先得注册一个 schema
D.register_schema(D.OpSchema("ErfInv", "custom", 1,
    inputs=[D.OpSchema.FormalParameter("X", "T")],
    outputs=[D.OpSchema.FormalParameter("Y", "T")],
    type_constraints=[("T", ["tensor(float)", "tensor(float16)"], "")]))
custom = onnxscript.values.Opset("custom", 1)

def erfinv_onnx(x):          # 坑 2：普通函数，不要加 @onnxscript.script
    return custom.ErfInv(x)

torch.onnx.export(model, inputs, path, dynamo=True,
    custom_translation_table={torch.ops.aten.erfinv.default: erfinv_onnx})
```

这条比 1a 难走，三个坑都是实测踩到的：

| 坑 | 症状 | 原因 |
|---|---|---|
| 没注册 schema | `No schema registered for 'ErfInv' version '1' and domain 'custom'` | onnxscript 的 `Opset.__getattr__` 要查 schema |
| 用了 `@onnxscript.script` | 报错指向**下一个**节点 `fc2` 的 `aten_linear` | 输出丢了 shape，下游才炸 —— 报错位置误导 |
| TRT parse 失败 | `Failed to import initializer` | 新导出器默认 `external_data=True`，权重写到 `.onnx.data`；`parser.parse(bytes)` 找不到它，要用 `parser.parse_from_file(path)` |

第三个特别阴险：报错看起来像 plugin 问题，其实和 plugin 无关。

两种导出器得到的 engine 性能一样（0.079 ms vs 0.079 ms），选哪个取决于你的
导出代码本来是哪种。

### 1c. 分解成标准算子：根本不要 plugin

同样用 symbolic 钩子，但这次生成的不是自定义节点，而是一串标准 ONNX 算子。
`erfinv` 用 Giles (2010) 的单精度近似 —— 两个 9 阶多项式加一个 `Where`：

```python
def erfinv_symbolic(g, x):
    xf = g.op("Cast", x, to_i=1)                       # fp32 里算
    w = Neg(Log((1 - xf) * (1 + xf)))
    p1 = horner(A, w - 2.5)                            # 中心区域
    p2 = horner(B, Sqrt(w) - 3.0)                      # 尾部
    return g.op("Cast", Where(w < 5, p1, p2) * xf, to_i=10)
```

（完整代码见 `exp_custom_plugin.py: export_decomposed`，26 行。）

**模型源码一行不改**，ONNX 只用 opset 17 的标准算子（导出后 73 个节点），
任何 TRT 版本都能 parse，**部署时没有 `.so` 要带**。

代价是数值：这是近似，不是 `erfinv` 本身。实测 cos 0.999997，和 plugin 的 0.999999
在同一量级；但如果你的算子没有好的近似，或者近似要上百个节点，这条路就不划算了。

## 10.3 逃生口 2：让 ONNX → engine 能过

### 只有 1a/1b 不够

第一个对照组：导出自定义节点，但**什么 plugin 都不注册**：

```
onnx parse failed: In node 4 with name: /ErfInv and operator: ErfInv
  (checkFallbackPluginImporter): INVALID_NODE:
  creator && "Plugin not found, are the plugin name, version, and namespace correct?"
```

`checkFallbackPluginImporter` 这个名字把机制说清楚了：parser 对不认识的 op_type
**不直接报错**，而是回退到"按名字查 plugin registry"（默认 version `"1"`、
namespace `""`，可以用节点属性 `plugin_version` / `plugin_namespace` 覆盖）。
查不到才报错。所以逃生口 1 和 2 的接口就是**一个字符串**。

### 2a. C++ plugin（IPluginV3）

`code/plugins/erfinv_plugin.cu`，170 行，**去掉注释和空行 120 行，其中 kernel 本身 10 行**：

```cpp
template <typename T>
__global__ void erfinvKernel(T const* in, T* out, int64_t n) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i < n) out[i] = static_cast<T>(erfinvf(static_cast<float>(in[i])));
}
```

其余 110 行是接口样板，每一块对应 builder 要问的一个问题：

| 方法 | builder 在问什么 |
|---|---|
| `getOutputShapes` | 输出 shape？（用 `DimsExprs`，动态维度也要能表达） |
| `getOutputDataTypes` | 输出 dtype？ |
| `supportsFormatCombination` | 这个 (位置, dtype, 布局) 组合你接受吗？ |
| `enqueue` | 在这个 stream 上跑 |
| `getFieldsToSerialize` | 序列化进 engine 的参数 |
| `clone` / `attachToContext` | 每个 execution context 一份实例 |
| `IPluginCreatorV3One` | 按名字造实例；`REGISTER_TENSORRT_PLUGIN` 静态注册 |

**`supportsFormatCombination` 值得多说一句**：这里接受 fp32 和 fp16 两种。如果只写
fp32，TensorRT 会在 plugin 两侧各插一个 fp16↔fp32 的 reformat 层 —— 不报错，只是慢。

注册方式：`.so` 一加载，静态构造函数就把 creator 塞进全局 registry。

```python
ctypes.CDLL("artifacts/liberfinv_plugin.so")    # Python builder
```
```bash
trtexec --onnx=m.onnx --fp16 --staticPlugins=liberfinv_plugin.so   # CLI
```

**部署耦合**：engine 里只存了"这里有个叫 ErfInv 的 plugin"，代码不在里面。
不带 `.so` 去加载：

```
IPluginRegistry::getCreator: Error Code 4: Cannot find plugin: ErfInv, version: 1
ICudaEngine::createExecutionContext: Cannot deserialize plugin since corresponding
  IPluginCreatorInterface not found in Plugin Registry
```

注意它在 **`createExecutionContext`** 才失败，不是在反序列化时 —— 健康检查如果只做
`deserialize_cuda_engine` 是查不出来的。另外 `.so` 是按 `-arch=sm_110a` 编译的，
**换 GPU 架构要重编**，这和 engine 本身的绑定是叠加的。

### 2b. Python plugin（`tensorrt.plugin`，QDP）

TensorRT 10 新增的 Python 装饰器 API，官方标注 **experimental**。两种写法：

**JIT 版（`@trtp.impl`）** —— 最好写，十几行：

```python
@trtp.register("custom::ErfInv")
def desc(x: trtp.TensorDesc) -> Tuple[trtp.TensorDesc]:
    return x.like()                       # 替代 getOutputShapes + getOutputDataTypes

@trtp.impl("custom::ErfInv")
def impl(x: trtp.Tensor, outputs: Tuple[trtp.Tensor], stream: int) -> None:
    xt, yt = torch.as_tensor(x, device="cuda"), torch.as_tensor(outputs[0], device="cuda")
    with torch.cuda.stream(torch.cuda.ExternalStream(stream)):
        erfinv_kernel[grid](xt, yt, xt.numel(), BLOCK=1024)   # Triton kernel
```

代价：**每次 enqueue TensorRT 都回调这个 Python 函数**。engine 离不开 Python 进程，
这条路线最大的部署优势（`.plan` 脱离 Python）没了。

**AOT 版（`@trtp.aot_impl`）** —— 返回 kernel 名 + PTX + launch 参数，PTX 嵌进 engine，
Python 只在构建期需要。实测 trtexec 能直接加载，**不需要任何 `.so`**。

这条踩了两个坑：

1. **NVIDIA 示例里用 Triton 生成 PTX，在这个环境下 segfault。** Triton 3.5 给每个
   kernel 的 PTX 签名额外追加了两个隐藏的 scratch 指针参数：

   ```
   .entry k1(.param .u64 k1_param_0, .param .u32 k1_param_1, .param .u64 k1_param_2,
             .param .u64 k1_param_3, .param .u64 k1_param_4)   ← 后两个是 Triton 加的
   ```

   TensorRT 10.14 的 AOT launcher 不传它们 —— **构建成功，第一次推理段错误**。
   本教程改用 `nvcc -ptx` 编译 `extern "C"` 的 CUDA C，签名就是你写的那样。
2. **只支持 PTX，不支持 cubin。** 换成 `nvcc -cubin` 会在 `onShapeChange` 失败。

## 10.4 实测

`customop`（fc1 → clamp → erfinv → fc2），FP16，每个变体在独立子进程里跑
（plugin registry 是进程全局的，否则前一个变体注册的 plugin 会污染后一个）。

两列延迟的区别：
- **median ms**：和第 5/6 章同一套 CUDA event 计时，包含 Python 侧调用开销
  （TRT 路线走的是第 3 章那个 `TrtRunner`）。
- **trtexec GPU ms**：同一个 `.plan` 用 trtexec 加载，只算 GPU 时间 ——
  相当于部署到 C++ 服务里能拿到的数字。这一列有值，本身就证明了 engine 能脱离 Python。

### b=32（与第 5/6 章同一配置）

| 变体 | 逃生口 | median ms | trtexec GPU ms | 脱离 Python | cos |
|---|---|---|---|---|---|
| eager | — | 0.065 | — | 否 | 0.999999 |
| **aoti** | — | **0.060** | — | **是** | 1.000000 |
| dynamo_trt（部分回退） | — | 0.201 | — | 否 | 0.999998 |
| onnx 自定义节点，无 plugin | 1a | **构建失败** | — | — | — |
| onnx + C++ plugin | 1a + 2a | 0.079 | 0.022 | **是** | 0.999999 |
| trtexec + C++ plugin | 1a + 2a | 0.090 | 0.021 | **是** | 0.999998 |
| onnx(dynamo 导出) + C++ plugin | 1b + 2a | 0.079 | 0.021 | **是** | 0.999999 |
| onnx + Python plugin JIT | 1a + 2b | 0.313 | — | 否 | 0.999998 |
| onnx + Python plugin AOT | 1a + 2b | 0.276 | 0.179 | **是** | 0.999999 |
| **onnx 分解成标准算子** | 1c | 0.088 | **0.019** | **是** | 0.999997 |
| dynamo_trt + C++ plugin converter | 2a | 0.118 | — | 否 | 0.999998 |

### b=8192（计算量放大 256 倍）

| 变体 | median ms | trtexec GPU ms |
|---|---|---|
| eager | 0.192 | — |
| **aoti** | **0.144** | — |
| dynamo_trt（部分回退） | 0.360 | — |
| onnx + C++ plugin | 0.233 | 0.179 |
| trtexec + C++ plugin | 0.228 | 0.186 |
| onnx(dynamo 导出) + C++ plugin | 0.234 | 0.191 |
| onnx + Python plugin JIT | 0.424 | — |
| onnx + Python plugin AOT | 0.511 | 0.356 |
| **onnx 分解成标准算子** | **0.200** | **0.159** |
| dynamo_trt + C++ plugin converter | 0.229 | — |

构建耗时都在 6–10 s（ONNX 导出 0.17 s，dynamo 导出器 2.0 s）；engine 0.57–0.59 MB。

## 10.5 读数据

### 为什么分解比手写 plugin 快：plugin 是融合屏障

b=8192 时 `trtexec --dumpProfile` 的逐层时间：

```
C++ plugin:                                 分解成标准算子:
  __myl_FcMaxMin         0.049 ms             __myl_Fc               0.047 ms
  /ErfInv  (plugin)      0.075 ms             __myl_MaxMinAddSubMulLog
  /fc2/Gemm              0.040 ms               NegLtSqrtAddMul...    0.050 ms
                                              /fc2/Gemm              0.040 ms
  合计                   0.166 ms             合计                   0.140 ms
```

左边：`clamp` 被融进了 fc1 的 epilogue，但 plugin 是 TensorRT 看不见内部的黑盒，
只能单独一个 kernel，读一遍写一遍 8 MB。右边：73 个标准算子（连同 clamp）被 Myelin
融成**一个** kernel。融合后的 kernel 比手写的 `erfinvf` 还快（0.050 vs 0.075 ms）。省掉的那次显存往返
是确定的；CUDA 的 `erfinvf` 实现可能也比 Giles 近似更重，但本教程没有单独测这一项。

> **规律：能用标准算子表达的，先别写 plugin。** plugin 的 kernel 再快，也拿不到
> TensorRT 的跨层融合。只有当标准算子表达不了（排序、自定义 attention、
> 稀疏/图操作）或者表达出来太慢时，plugin 才值得。

### Python AOT plugin 为什么慢 40 倍

b=32 和 b=8192 下，AOT plugin 那一层都在 0.16–0.18 ms，**和数据量无关** ——
说明是固定开销。把 kernel 换成 `y[i] = x[i]`（什么都不算），这一层还是 0.155 ms。

用 nsys 看 200 次推理的 CUDA API：

```
                          C++ plugin    Python AOT plugin
cuModuleLoadData 调用数        2              202
cuModuleLoadData 总耗时      0.32 ms        36.7 ms
```

**AOT plugin 每次 enqueue 都重新加载一次 PTX 模块**（每次约 0.18 ms，很可能主要是
PTX→SASS 的 JIT 编译）。这是 TensorRT 10.14 的实现行为，不是你的代码问题，也绕不过去
（cubin 不支持）。在 API 脱离 experimental 之前，**别把 QDP AOT 用在热路径上**。

### b=32 时 median 和 GPU 时间差了 3–4 倍

ONNX+C++ plugin：median 0.079 ms，GPU 时间 0.022 ms。差的 0.057 ms 是 `TrtRunner`
每次调用的 Python 开销（`torch.empty` 分配输出、`set_tensor_address`、stream 同步）。
这个模型小到 Python 开销是主体。

所以对这个模型要分两种情况读：
- **Python 服务里**：TRT 路线最好也就 0.079 ms，**比 eager（0.065）还慢**，
  AOTI（0.060）最快。第 5 章的结论不变。
- **C++ 服务里**：engine 只要 0.02 ms。但这时 eager 也该换成 CUDA graph 来比
  （实测 eager + CUDA graph replay 0.015 ms），AOTI 在 C++ 里同样没有 Python 开销。
  **小模型上编译不编译差别不大，第 6 章规律 1 依然成立。**

### Torch-TensorRT 也能用同一个 plugin

```python
@dynamo_tensorrt_converter(torch.ops.aten.erfinv.default, supports_dynamic_shapes=True)
def erfinv_converter(ctx, target, args, kwargs, name):
    creator = trt.get_plugin_registry().get_creator("ErfInv", "1", "")
    plugin = creator.create_plugin("ErfInv", trt.PluginFieldCollection([]),
                                   trt.TensorRTPhase.BUILD)
    layer = ctx.net.add_plugin_v3([get_trt_tensor(ctx, args[0], name)], [], plugin)
    return layer.get_output(0)
```

注册之后 `dynamo_trt` 的子图计数从 **2 TRT / 1 回退** 变成 **1 TRT / 0 回退**，
延迟 0.201 → 0.118 ms（b=32）、0.360 → 0.229 ms（b=8192）。

这说明第 4 章的"部分回退"和本章的 plugin **不是二选一**：部分回退让你先跑起来，
等 profiling 证明回退子图是瓶颈，再写 plugin 把它收回 engine。**plugin 是两条 TRT
路线共用的资产**，`.so` 写一次两边都能用。

（Torch-TensorRT 2.9 还有 `torch_tensorrt.dynamo.conversion.plugins.custom_op`，
能从 `torch.library` 自定义算子自动生成 QDP plugin 和 converter。第 4 章 4.5 实测了：
0 回退，但 b=32 下 0.56–0.69 ms，比部分回退还慢 —— 10.5 测到的 QDP 开销同样适用。
那一节也给出了用户自定义算子（而不是 aten 算子）接这个 C++ plugin 的完整写法。）

## 10.6 修正后的对比：遇到不支持的算子

把第 5 章那张表加上本章的方案：

| 方案 | 工程成本 | 部署要带什么 | customop b=32 | b=8192 | 脱离 Python |
|---|---|---|---|---|---|
| **aoti** | 零 | `.pt2` | **0.060** | **0.144** | 是 |
| eager | 零 | 整个 PyTorch | 0.065 | 0.192 | 否 |
| **ONNX 分解成标准算子** | 中：写一个 symbolic，**要有好的数学近似** | `.plan` | 0.088 | 0.200 | 是 |
| ONNX + C++ plugin | 高：120 行 C++，要会 TRT plugin 接口 | `.plan` + **`.so`（按架构编译）** | 0.079 | 0.233 | 是 |
| dynamo_trt + C++ plugin | 高：同一个 `.so` + 10 行 converter | GraphModule + `.so` | 0.118 | 0.229 | 否（默认） |
| ONNX + Python plugin AOT | 低：30 行 Python | `.plan` | 0.276 | 0.511 | 是 |
| ONNX + Python plugin JIT | 最低：15 行 Python | `.plan` + Python 进程 | 0.313 | 0.424 | 否 |
| dynamo_trt（部分回退） | 零 | GraphModule | 0.201 | 0.360 | 否 |
| ONNX，什么都不做 | — | — | 构建失败 | 构建失败 | — |

## 10.7 这改变了前面哪些结论

| 前面的说法 | 修正 |
|---|---|
| 第 0/3/4 章："ONNX→TRT 全有或全无，要继续只能写 C++ plugin" | **有两个逃生口**，而且最好的那个（分解成标准算子）连 plugin 都不用写 |
| 第 5 章："customop 上唯一能脱离 Python 的是 AOTI" | ONNX + plugin / 分解 也能脱离 Python。**但 AOTI 仍然最快**，这条不变 |
| 第 4 章："部分回退是 R4 相对 ONNX 最实质的能力差异" | 能力差异在于**零工程成本就能跑**；ONNX 花点工程成本也能跑，而且**能跑得比部分回退快**（0.079 vs 0.201） |
| 第 6 章决策流程："`torch.onnx.export` 不能过 → dynamo_trt" | 中间该插一步：**先看能不能在导出期分解成标准算子** |

## 10.8 选型建议

遇到 TensorRT 不支持的算子时，按这个顺序试：

```
1. 换 AOTInductor 试一下         零成本。这个模型上它就是最优解。
   └─ 必须用 TensorRT ↓
2. 导出期分解成标准 ONNX 算子    不碰 C++，不带 .so，能被 TRT 融合。
   └─ 表达不了 / 太慢 ↓
3. dynamo_trt 部分回退先跑起来    零成本，看子图计数和 profile，确认这个算子真的是瓶颈。
   └─ 回退子图是瓶颈 ↓
4. 写 C++ IPluginV3              一份 .so，ONNX 路线（parser）和 dynamo 路线（converter）都能用。
                                 记得 supportsFormatCombination 接受 fp16。

Python plugin（QDP）：拿来验证 plugin 接口和数值没问题，不要上热路径（TRT 10.14）。
```

## 10.9 本章要记住的三件事

1. **ONNX 路线遇到不支持的算子，不是死路，是两个逃生口。** 导出侧和构建侧通过一个
   字符串（op_type = plugin 名）对接。
2. **plugin 是融合屏障。** 能用标准算子表达的先分解；手写 kernel 再快也拿不到
   TensorRT 的跨层融合。
3. **plugin 修好的是"能不能"，不是"快不快"。** 在这个模型上 AOTI 依然最快 ——
   和第 5 章一样，"能转"不等于"该转"。

---

← [第 9 章：动态 shape](09_dynamic_shapes.md) | [返回 README](README.md)
