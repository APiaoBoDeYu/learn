# 第 1 章：环境搭建 —— 六条路线的依赖各自是什么

本章目标：把六条路线的运行环境全部装好，并且说清楚**每条路线真正依赖什么**。
这件事值得单独一章，因为"路线的依赖成本"本身就是选型的一部分：一条需要额外装
1.5 GB 依赖、还要和 TensorRT 版本对齐的路线，和一条 PyTorch 自带的路线，即使
性能相同也不是同一个选择。

## 1.1 本教程的实测环境

```
硬件     : NVIDIA Thor (Jetson, aarch64), compute capability (11, 0) / sm_110a
CUDA     : 13.0
cuDNN    : 9.17.0
PyTorch  : 2.9.1
TensorRT : 10.14.1.48
torchvision : 0.24.1
```

这是一块**边缘推理平台**，不是数据中心 GPU。这个事实会贯穿整个教程。关于构建时间
有一件事值得提前说清楚，因为我在准备教程时先被它骗了一次：

第一次跑 resnet18 的 TensorRT 构建时，进程跑了 **40 分钟还没结束**，我一度准备把
"Jetson 上 engine 构建极慢"写进教程当结论。后来在干净环境下重跑，同一个模型
**17～21 秒**就构建完成（第 6 章的实测表格里可以看到）。差异的来源是那次构建
和另一个占满 GPU 的进程抢资源。

教训值得记下来，因为它对任何 benchmark 都成立：

> **TensorRT 构建期会吃满 GPU。同时跑两个构建，或者一边构建一边测延迟，
> 得到的数字毫无意义 —— 而且慢得会让你误以为是平台特性。**

本教程 `code/run_bench.py` 因此**严格串行**执行所有路线，每条之间 `empty_cache()`
+ 同步 + sleep。

统一用 `python3`，所有代码在 `/root/data/learn/torch2trt/code/` 下。

## 1.2 各路线的依赖矩阵

| 路线 | 需要额外装什么 | 额外体积 | 版本对齐风险 |
|---|---|---|---|
| R0 eager | 无（PyTorch 自带） | 0 | 无 |
| R1 torch.compile | 无（PyTorch 自带 Inductor + Triton） | 0 | 低（triton 随 torch 走） |
| R2 onnx → TRT (Python) | `onnx` + `tensorrt` | ~1.5 GB | **中**：opset 版本 ↔ TRT 版本 |
| R3 trtexec | `tensorrt`（提供 `trtexec` 二进制） | 同上 | 同上 |
| R4 dynamo_trt | `torch-tensorrt` + `tensorrt` | ~1.5 GB + 3.5 MB | **高**：torch ↔ torch-tensorrt ↔ tensorrt 三方对齐 |
| R5 ts_trt | 同 R4（同一个包） | 同上 | 同上 |
| R6 AOTInductor | 无（PyTorch 自带）+ 需要系统 C++ 编译器 | 0 | 低 |
| 量化（第 8 章） | `nvidia-modelopt` + `onnx_graphsurgeon` + `onnxscript` + `huggingface_hub` | ~200 MB | 中 |

这张表已经能给出一个结论：**R1 和 R6 是零依赖路线**。如果它们的性能对你够用，
整条 TensorRT 依赖链（以及它带来的版本地狱）都可以不引入。这就是第 6 章要用实测
数据回答的问题。

## 1.3 验证 PyTorch 与 TensorRT

先确认基础环境：

```bash
python3 -c "
import torch, tensorrt
print('torch   ', torch.__version__)
print('tensorrt', tensorrt.__version__)
print('device  ', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
"
```

实际输出：

```
torch    2.9.1
tensorrt 10.14.1.48
device   NVIDIA Thor (11, 0)
```

`trtexec`（R3 路线要用的 CLI）本机在 `/usr/bin/trtexec`。注意它**不是** pip 装
`tensorrt` 时装进 venv 的，而是随系统 TensorRT 包走的。在容器镜像里它经常在
`/usr/src/tensorrt/bin/trtexec`，如果 `which trtexec` 找不到，先去那里看看：

```bash
which trtexec || ls /usr/src/tensorrt/bin/
```

## 1.4 安装 Torch-TensorRT（R4 / R5 路线）

这是本章唯一有坑的一步，而且坑很有代表性，值得完整记录。

### 第一步：确认有 aarch64 wheel

```bash
pip3 index versions torch-tensorrt
```

```
torch-tensorrt (2.14.0)
Available versions: 2.14.0, 2.13.0, 2.12.1, 2.12.0, 2.11.0, 2.10.0, 2.9.0, 2.8.0
```

**版本号选择规则**：Torch-TensorRT 的主次版本号跟 PyTorch 对齐。本机
`torch==2.9.1`，所以选 `torch-tensorrt==2.9.0`。不要直接装最新的 2.14.0，
它会把你的 torch 一起升级掉。

```bash
pip3 install --no-deps torch-tensorrt==2.9.0
```

`--no-deps` 是关键：不加它，pip 会因为依赖声明而尝试改动 torch 和 tensorrt 的版本。
代价是你要自己补缺失的依赖（见下一步）。

### 坑 1：缺少 `dllist`

装完直接 import 报错：

```
ModuleNotFoundError: No module named 'dllist'
```

来自 `torch_tensorrt/_utils.py` 里的 `check_cross_compile_trt_win_lib()` —— 一个
**检测 Windows 交叉编译能力**的函数。在 Linux 上这个检查的结果毫无意义，但它在
模块导入时无条件执行，所以缺这个包会直接让 import 失败。这是 `--no-deps` 的代价：

```bash
pip3 install dllist
```

### 坑 2：TensorRT 版本"不兼容"警告

```
torch-tensorrt 2.9.0 requires tensorrt<10.14.0,>=10.13.0,
but you have tensorrt 10.14.1.48 which is incompatible.
```

`torch-tensorrt==2.9.0` 的元数据把 TensorRT 钉在 `>=10.13.0,<10.14.0`，而本机是
10.14.1。**实测这个警告可以忽略**——10.13 到 10.14 之间 Torch-TensorRT 用到的
Python API 没有破坏性变更，后面所有实验都正常跑通了。

但请注意这是**实测结论，不是通用保证**。Torch-TensorRT 通过 C++ ABI 和 TensorRT
耦合，跨越更大的版本差（比如 TRT 9 vs 10）会真的崩。判断方法很简单，就是下一步的
冒烟测试：能编译出 engine 并且数值对得上，就说明这个组合可用。

### 坑 3：三条 import 期警告

import 时会固定打三行：

```
Unable to import quantization op. Please install modelopt library ...
Unable to import quantize op. Please install modelopt library ...
TensorRT-LLM is not installed. Please install TensorRT-LLM or set TRTLLM_PLUGINS_PATH ...
```

含义分别是：
- 前两条：没装 `nvidia-modelopt`，所以**无法编译量化（INT8/FP8）模型**。
  第 1～7 章的实验都是 FP16，不受影响；**第 8 章要做量化，那里会装上它**（见 1.7）。
- 第三条：没装 TensorRT-LLM，所以 `torch.distributed` 相关算子没有 converter。
  单卡推理不受影响。

三条都是"功能缺失提示"，不是错误。本教程 `code/common.py` 里设了
`TORCH_TENSORRT_LOG_LEVEL=ERROR` 让输出干净一些。

### 验证安装

```bash
python3 -c "
import torch, torch_tensorrt as tt
print('ttrt', tt.__version__)
print(tt._features._enabled_features_str())
"
```

实际输出：

```
ttrt 2.9.0
Enabled Features:
 - Dynamo Frontend: ENABLED
 - Torch-TensorRT Runtime: ENABLED
 - FX Frontend: ENABLED
 - TorchScript Frontend: ENABLED
 - Refit: ENABLED
 - QDP Plugin: ENABLED
 - TensorRT-RTX: DISABLED
```

`Dynamo Frontend: ENABLED` 和 `TorchScript Frontend: ENABLED` 分别对应本教程的
R4 和 R5 路线，两条都可用。

## 1.5 冒烟测试：真的能编出 engine 吗

import 成功不等于能用。必须跑一次真实编译：

```python
import torch, torch_tensorrt as tt

m = torch.nn.Sequential(
    torch.nn.Conv2d(3, 16, 3, padding=1), torch.nn.ReLU(),
    torch.nn.Conv2d(16, 16, 3, padding=1), torch.nn.ReLU(),
).eval().cuda().half()
x = torch.randn(1, 3, 64, 64, device="cuda", dtype=torch.half)

ep = torch.export.export(m, (x,))
trt = tt.dynamo.compile(ep, inputs=[x], enabled_precisions={torch.half})
print("maxerr", (m(x) - trt(x)).abs().max().item())
```

实际输出：

```
WARNING:torch_tensorrt.dynamo._compiler:4 supported operations detected in subgraph
  containing 4 computational nodes. Skipping this subgraph, since min_block_size
  was detected to be 5
maxerr 0.0
```

**这里有一个非常容易骗过自己的陷阱**：`maxerr 0.0` 看起来完美，但那个 WARNING 说明
TensorRT **一个 engine 都没建**。`min_block_size` 默认是 5，这个模型只有 4 个算子，
所以整张图被跳过、原样退回 PyTorch 执行。误差当然是 0 —— 因为跑的就是原模型。

这是本教程第一个重要教训：

> **验证 TensorRT 转换是否生效，不能只看数值误差。误差为 0 很可能意味着根本没转。**

要真正建 engine，得把 `min_block_size` 调低（或者用更大的模型）：

```python
trt = tt.dynamo.compile(ep, inputs=[x], enabled_precisions={torch.half},
                        min_block_size=1)
```

本教程 `code/routes.py` 里的 `build_dynamo_trt` 因此默认 `min_block_size=1`，
并且会数出 TRT 子图和 PyTorch 回退子图的数量记进结果里 —— 这样"转了多少"是一个
被记录的数字，不是一个假设。第 5 章会用这两个计数来量化部分回退的代价。

### 坑 4：Thor 上的显存告警

同一次编译还会打：

```
WARNING: Remaining GPU memory may not be enough to compile the TensorRT engine
for this model resulting in an OOM error, Consider setting offload_module_to_cpu=True
```

Jetson 是**统一内存**架构（CPU 和 GPU 共用这 120 GB），TensorRT builder 在构建期
需要同时持有 PyTorch 的权重和自己的工作空间，所以它保守地告警。按提示加上
`offload_module_to_cpu=True` 让 builder 在构建期把权重暂存到 host 侧即可，
`routes.py` 里已经默认开启。

> **⚠ 但这个参数有个副作用，第 9 章踩到了：它会把你的模型原地搬到 CPU 并且
> 不搬回来。** 之后任何对这个 model 的使用都会报 device 不匹配，而且报错出现在
> 完全无关的后续代码里。`routes.py` 现在记录原始 device 并在 `finally` 里恢复，
> 详见 9.5。

## 1.6 关于本教程不涉及的依赖

为了让边界清楚，明确列出**没有装、因此教程里对应内容是"路线说明"而非实测**的部分：

- ~~`nvidia-modelopt`：INT8/FP8 量化~~ → **第 8 章已装并实测**，安装方式见 1.8。
- `onnxruntime`：如果你只是想加速 ONNX 而不上 TensorRT，ORT 是一条独立路线。
  本教程不覆盖（它不是"转 TensorRT"）。
- `polygraphy`：ONNX 调试工具链。第 7 章会说明它能解决什么问题。
  （`onnx-graphsurgeon` 第 8 章作为 modelopt 的依赖装上了。）
- `tensorrt-llm`：LLM 专用推理栈。第 7 章讲它和本教程六条路线的关系（它是**另一个层级**
  的东西，不是第七条路线）。

## 1.7 安装量化依赖（第 8 章用）

第 8 章的量化实验需要 `nvidia-modelopt`。它的依赖链有三个"装了主包还缺东西"的坑，
按顺序装完即可：

```bash
pip3 install nvidia-modelopt      # 主包
pip3 install huggingface_hub      # modelopt.torch 导入期无条件 import 它
pip3 install onnx_graphsurgeon    # modelopt.onnx 需要
pip3 install onnxscript           # torch.onnx 的 dynamo 导出器需要
```

三个附加包的共同点：**主包的依赖声明没覆盖它们，但导入时会无条件用到**，
所以只装主包会在第一次 import 或第一次导出时失败。验证：

```bash
python3 -c "
import torch, modelopt.torch.quantization as mtq
print([c for c in dir(mtq) if c.endswith('_CFG')][:6])
"
```

实际输出（本机 modelopt 0.47.0）：

```
['FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG', 'FP8_AFFINE_KV_CFG', 'FP8_DEFAULT_CFG',
 'FP8_KV_CFG', 'FP8_PER_CHANNEL_PER_TOKEN_CFG', 'INT4_AWQ_CFG']
```

另外确认这块 GPU 真的有 INT8 加速能力（不然第 8 章的实验没有意义）：

```bash
python3 -c "
import tensorrt as trt
b = trt.Builder(trt.Logger(trt.Logger.ERROR))
print('fast_int8', b.platform_has_fast_int8)
print('fast_fp16', b.platform_has_fast_fp16)
"
```

本机两个都是 `True`。注意 `platform_has_fast_int8=True` 只说明**有 INT8 tensor
core**，不保证**每个算子都有高效的 INT8 tactic** —— 第 8 章 FP8 那一节就是这个
区别的反面教材。

## 1.8 环境自检脚本

```bash
cd /root/data/learn/torch2trt/code
python3 common.py      # 打印环境 + 跑一次 matmul 计时
python3 models.py      # 构建全部四个模型，确认都能前向
```

`common.py` 实际输出：

```
torch      : 2.9.1
tensorrt   : 10.14.1.48
torch_trt  : 2.9.0
cuda avail : True
device     : NVIDIA Thor
capability : (11, 0)
--------------------------------------------------------------------
matmul bench: {'median_ms': 0.975, 'p90_ms': 1.199, 'mean_ms': 1.006}
```

`models.py` 实际输出：

```
resnet18   in=[(8, 3, 224, 224)] out=(8, 1000) params=11.69M dtype=torch.float16
bert_tiny  in=[(8, 128)] out=(8, 2) params=4.77M dtype=torch.float16
ctrlflow   in=[(32, 512)] out=(32, 10) params=0.53M dtype=torch.float16
customop   in=[(32, 512)] out=(32, 10) params=0.27M dtype=torch.float16
```

两个脚本都跑通，就可以进第 2 章了。

---

← [第 0 章：全局地图](00_overview.md) | [第 2 章：基准与模型设计](02_baseline_and_models.md) →
