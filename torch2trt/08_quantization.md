# 第 8 章：量化 —— INT8 / FP8 到底值不值

第 6 章把量化列为本教程最大的覆盖缺口，并且说"如果你的场景要 INT8，
TensorRT 路线的分数应该明显上调"。本章装上 `nvidia-modelopt` 把这个缺口补掉。

结果比那句话复杂得多，而且**推翻了它**。

## 8.1 为什么量化必须换实验方法

前七章用 `weights=None`（随机权重）+ 随机输入，测延迟和路线间数值一致性够用。
量化不行，有两个独立原因：

**原因 1：随机权重的模型没有"准确率"可损失。** 量化的核心问题是"掉了几个点"，
一个随机初始化的 resnet18 准确率本来就是 0.1%，量化前后都是噪声。

**原因 2（更关键）：INT8 校准测的就是激活值分布。** 校准器的工作是统计每个张量的
激活值范围，然后选一组量化 scale。**拿随机噪声去校准，得到的 scale 对任何真实图片
都是错的。** 这不是精度问题，是方法论错误 —— 而且它会给你一个看起来正常的 engine。

所以本章换成：
- **`weights="IMAGENET1K_V1"`**：torchvision 的真实 ImageNet 预训练权重。
- **imagenette2-160**：ImageNet 的 10 类真实子集，9469 训练 / 3925 验证图片（约 99 MB）。
  因为它是真 ImageNet 的子集，预训练的 1000 类分类头**不用改**，直接读那 10 个类
  对应的 logit 做 argmax 就行。改分类头会把"量化损失"和"重训损失"混在一起。
- **校准集 128 张真实训练图片**，所有量化路线**共用同一份** —— 否则某条路线可能
  只是运气好抽到了更容易的图。

一个容易忽略的细节，`data.py` 里的：

```python
if shuffle_seed is not None:
    order = torch.randperm(len(items), generator=g).tolist()
```

图片在磁盘上**按类别分目录存放**。不打乱直接取前 128 张，校准集里只有
tench 和 English springer 两个类，校准器算出的激活范围来自 10 类里的 2 类。
固定种子打乱，既有代表性又可复现。

FP16 基线（800 张验证图）：**top-1 = 95.12%**。这是所有量化路线要对标的数字。

## 8.2 三种量化机制（差别比精度标签更重要）

| 机制 | 校准在哪做 | 图里有 Q/DQ 节点吗 | 需要额外包吗 |
|---|---|---|---|
| **TRT 原生校准器** | TensorRT 内部（C++ 回调） | 没有，scale 存在 engine 里 | 不需要 |
| **modelopt Q/DQ** | PyTorch 侧 | **有**，scale 烧进图 | `nvidia-modelopt` |
| **weight-only** | 只量化权重 | 有（只在权重上） | `nvidia-modelopt` |

### 机制 1：TensorRT 自己的熵校准器

```python
class Calibrator(trt.IInt8EntropyCalibrator2):
    def get_batch(self, names):
        if self.idx >= len(self.data):
            return None
        self._current = self.data[self.idx].float().contiguous()
        self.idx += 1
        return [int(self._current.data_ptr())]
```

TensorRT 用回调驱动：反复调 `get_batch()` 直到返回 `None`，每次用 FP32 跑一遍网络
收集激活值直方图，然后选一组让信息损失最小的 scale（这就是类名里的 "entropy"）。

**`self._current` 那一行不是多余的。** `get_batch` 返回的是**裸设备指针**，
TensorRT 不会帮你保持那块内存存活。如果 tensor 在返回后被 Python GC 回收，
TensorRT 就在已释放的显存上做校准 —— **静默地**产出一组看起来合理但完全错误的 scale。
这和第 3 章 `TrtRunner` 里那个 `self._keep` 是同一类问题：手写 TensorRT 运行时
你要自己管生命周期。

### 机制 2：modelopt 插入 Q/DQ 节点

```python
qmodel = mtq.quantize(model, mtq.INT8_DEFAULT_CFG, forward_loop)
```

modelopt 在 PyTorch 侧跑校准，然后**改写模型**，插入显式的 quantize/dequantize 算子。
实测在 resnet18 上插入了 **107 个 quantizer**，导出 ONNX 后有 **44 个
QuantizeLinear/DequantizeLinear 节点**。

scale 烧进图里，所以下游任何编译器都能看见 —— 这是它比机制 1 更可移植的原因，
也是拿到 FP8 的唯一途径。

`quant_routes.py` 里加了一个计数函数，理由和第 4 章数 TRT 子图一样：

```python
def _count_qdq(onnx_path) -> int:
    """Q/DQ 节点数为 0 意味着你建了个普通 FP16 engine，INT8 flag 什么都没做。"""
```

**把这个数字当标准检查项。** "量化到底生效了吗"必须是一个可读的数字，不是假设。

## 8.3 一个意外发现：量化工具自己踩了第 5 章的坑

我原计划走 `modelopt → torch.export → Torch-TensorRT`（和第 4 章的 R4 一致）。
两条路都失败了，而**失败原因很有教育意义**：

**FP32 Q/DQ 模型**：`torch.export(strict=False)` 成功，但 Torch-TensorRT 编译时报：

```
RuntimeError: Cannot access data pointer of Tensor (e.g. FakeTensor,
FunctionalTensor). If you're using torch.compile/export/fx, it is likely that we
are erroneously tracing into a custom kernel.
```

modelopt 的 fake-quant 算子调用了自己的 CUDA 扩展（日志里能看到
`Loading extension modelopt_cuda_ext`），不能被 trace 进去。

**FP16 Q/DQ 模型**：失败得更早，而且眼熟：

```
GuardOnDataDependentSymNode: Could not guard on data-dependent expression Eq(u0, 1)
```

导出的图指向根因：

```python
# modelopt/torch/quantization/tensor_quant.py:375
min_1: "f32[]" = torch.ops.aten.min.default(to_2)
lt:   "b8[]"  = torch.ops.aten.lt.Scalar(min_1, 0)
ne:   "b8[]"  = torch.ops.aten.ne.Scalar(lt, 0)
item: "Sym(Eq(u0, 1))" = torch.ops.aten.item.default(ne)
```

**modelopt 的量化器内部有一句 `if amax.min() < 0`** —— 就是第 5 章那个数据相关控制流
模式，一模一样。NVIDIA 的量化工具撞在了它自己要警告用户的那面墙上。

这不是抱怨，是一个有用的观察：**第 5 章那条结论的适用范围比"你的模型"更广，
它同样适用于你依赖的库。** 一个库只要内部有数据相关分支，就会限制你能用哪条导出路线。

**能用的路是 NVIDIA 官方文档的那条**：用**传统（trace）ONNX 导出器**把 Q/DQ 节点
序列化出来，再交给 TensorRT builder 折叠成 INT8/FP8 层。注意必须 `dynamo=False`
（dynamo 导出器翻译不了 modelopt 的 fake-quant 算子，在第 3/3 步报 `ConversionError`）。
这条路上 TensorRT **不需要校准器** —— scale 已经在图里了，这正是 Q/DQ 的意义。

## 8.4 实测结果（batch=8）

```
| route | precision | median ms | vs fp16 eager | vs fp16 TRT | top1 | build s |
|---|---|---|---|---|---|---|
| eager | fp16 | 1.944 | 1.00x | 0.22x | 95.12% | 0.0 |
| dynamo_trt | fp16 | 0.423 | 4.60x | 1.00x | 95.12% | 8.2 |
| onnx_trt | fp16 | 0.410 | 4.75x | 1.03x | 95.12% | 17.5 |
| modelopt_int8_eager | int8 | 7.402 | 0.26x | 0.06x | 94.88% | 0.5 |
| trt_int8_calib | int8+fp16 | 0.450 | 4.32x | 0.94x | 95.00% | 29.0 |
| modelopt_int8 | int8 | 0.433 | 4.49x | 0.98x | 95.00% | 23.6 |
| modelopt_int8_wo | int8_wo | 0.714 | 2.72x | 0.59x | 94.88% | 21.8 |
| modelopt_fp8 | fp8 | 0.662 | 2.94x | 0.64x | 94.75% | 125.4 |
```

准确率变化（相对 FP16 的 95.12%）：

| route | top1 | delta |
|---|---|---|
| trt_int8_calib | 95.00% | **-0.13 pp** |
| modelopt_int8 | 95.00% | **-0.13 pp** |
| modelopt_int8_wo | 94.88% | -0.25 pp |
| modelopt_fp8 | 94.75% | -0.38 pp |

**先说好消息：准确率损失非常小。** INT8 掉 0.13 个百分点（800 张里错 1 张），
FP8 掉 0.38。对绝大多数应用这是可以接受的。两种 INT8 机制给出**完全相同**的
准确率，说明 TRT 熵校准和 modelopt 的默认校准在这个模型上等价。

**坏消息：`vs fp16 TRT` 那一列全部 ≤ 1.00x。没有一条量化路线比 FP16 快。**

（复核：我把整个实验重跑了一遍，准确率数字**逐位相同**，延迟在 3% 内
（`trt_int8_calib` 0.450 → 0.413，仍是 0.99x），
"batch=8 下量化不赚"这个结论稳定。）

## 8.5 为什么 INT8 没有更快 —— 以及一个必须做的追加实验

这个结果和普遍预期相反，所以**不能直接写成结论**。最可能的无罪解释是：
**batch=8 的 resnet18 太小，没有把 tensor core 喂饱**。这个尺寸下延迟可能由
kernel 启动开销和 INT8 在每个精度边界插入的 reformat kernel 主导 —— 两者都是固定成本，
量化只会让它变差。

如果这个解释对，INT8 应该随 batch 增大而反超。如果 batch=64 时 INT8 还是输，
那结论就是关于这块硬件的，不是关于测量方法的。

这就是"我的 benchmark 说 X"和"X 是真的"之间的区别：**对最可能的混淆因素做一次
对抗性追加实验。** `exp_quant_batch.py`：

```
| batch | onnx_trt fp16 (ms) | modelopt_int8 (ms) | modelopt_fp8 (ms) | int8 vs fp16 | fp8 vs fp16 |
|---|---|---|---|---|---|
| 1 | 0.282 | 0.315 | 0.352 | 0.89x | 0.80x |
| 8 | 0.413 | 0.424 | 0.654 | 0.97x | 0.63x |
| 32 | 0.949 | 0.807 | 1.782 | **1.18x** | 0.53x |
| 64 | 1.484 | 1.300 | 3.352 | **1.14x** | 0.44x |
```

**假设被证实了。INT8 在 batch ≥ 32 时反超 FP16，加速 1.14–1.18x。**

每张图延迟（吞吐视角）：

| batch | fp16 | int8 |
|---|---|---|
| 1 | 0.2822 | 0.3154 |
| 8 | 0.0516 | 0.0530 |
| 32 | 0.0297 | **0.0252** |
| 64 | 0.0232 | **0.0203** |

所以正确的结论是有条件的：

> **INT8 的收益需要足够大的 batch 才能兑现。** 小 batch（延迟敏感的在线服务）下
> INT8 反而更慢，因为 reformat 开销是固定的而计算量太小。大 batch（离线批处理、
> 高吞吐服务）下 INT8 给出 1.14–1.18x。

**如果我只测了 batch=8 就下结论，会得到一个错误的结论。** 这是本教程第二次靠
追加实验避免错误结论（第一次是第 1 章那个 40 分钟构建）。

## 8.6 FP8：在这块硬件上是净损失

FP8 的趋势和 INT8 **相反**，而且越来越糟：0.80x → 0.63x → 0.53x → 0.44x。
batch 越大越差，到 batch=64 时比 FP16 慢一倍多。

解释：**这块 GPU（Thor, sm_110a）的 TensorRT 10.14 缺少高效的 FP8 卷积 tactic。**
Q/DQ 节点确实在图里、engine 确实建出来了、数值也对（94.75%），但 TensorRT
找不到好的 FP8 卷积实现，只能在 FP8 和 FP16 之间反复 reformat。batch 越大，
这些 reformat 搬运的数据越多，所以劣势越明显。

（我一开始还想把"FP8 构建耗时 125.4s，是 INT8 的 5 倍"当作佐证，说 builder
花了大量时间找 tactic。**复核时这个数字塌了**：重跑时 FP8 构建只要 22.1s，
和 INT8 持平 —— 125.4s 是冷 timing cache 的一次性成本，不是 FP8 的特性。
延迟结论不受影响，但这条"佐证"是错的，删掉。）

> **FP8 是硬件/版本高度相关的。** 它在 Hopper/Blackwell 数据中心卡 + 新版 TensorRT
> 上（特别是对 transformer 的 matmul）是重要优化，但不要假设它在你的卡上就有收益 ——
> 实测 5 分钟就能知道。

## 8.7 `modelopt_int8_eager`：一个用来定位问题的对照行

`modelopt_int8_eager` 是 7.402ms —— **比 FP16 eager 慢 3.8 倍，比 FP16 TRT 慢 18 倍**。

这一行不是为了跑得快，它是个诊断工具。它把两件容易混淆的事分开：

- **量化对准确率的影响**（94.88%，可以和编译后的路线比）
- **量化对延迟的影响**（需要编译器支持才能兑现）

Q/DQ 模型在 eager 下必然慢：它插入了显式的 quantize/dequantize 算子，
而 PyTorch eager **没有任何 kernel 能利用这些信息** —— 纯粹是白加的开销。

有了这一行，当你的编译 INT8 路线也很慢时，你能判断是量化本身的问题还是编译的问题。
**做量化实验时建议永远保留这一行。**

## 8.8 weight-only INT8：一个关于瓶颈的教训

`modelopt_int8_wo` 只量化权重、激活值保持 FP16。结果 0.714ms vs FP16 的 0.410ms ——
**慢了 74%**。

原因：weight-only 是**显存带宽优化**。它把权重体积减半，收益出现在
**权重读取是瓶颈**的时候 —— 典型场景是 LLM 的 decode 阶段（batch=1，
每个 token 都要把整个权重矩阵读一遍，算术强度极低）。

resnet18 是**计算密集**的卷积网络，权重读取根本不是瓶颈。所以 weight-only
拿不到任何收益，还要付反量化的开销。

> **量化类型必须匹配瓶颈类型。** weight-only 对 LLM 有意义，对 CNN 没有。
> 先搞清楚你是 memory-bound 还是 compute-bound，再选量化方案。

## 8.9 对第 6 章选型表的修正

第 6 章说"如果你的场景要 INT8，TensorRT 路线的分数应该显著上调"。
**这句话需要限定条件**：

| 场景 | 量化建议 | 依据 |
|---|---|---|
| 在线服务，batch 1–8，延迟敏感 | **不要量化**，用 FP16 | INT8 在 b=1/8 是 0.89x/0.97x |
| 批处理/高吞吐，batch ≥ 32 | **INT8 值得做**，1.14–1.18x + 掉 0.13pp | b=32/64 实测 |
| CNN | INT8 全量化；**不要** weight-only | weight-only 0.59x |
| LLM / memory-bound | weight-only 或 INT4 | 本教程没测，见第 7 章 |
| FP8 | **先实测再相信** | 本硬件上 0.44–0.80x，净损失 |
| 显存受限而非延迟受限 | 量化仍然有用（engine 更小） | 见下 |

注意最后一行：即使延迟没改善，量化仍然能减小**权重显存占用**。
如果你的问题是"模型放不进显存"而不是"跑得不够快"，那量化的价值完全不同。

## 8.10 动手

```bash
cd /root/data/learn/torch2trt/code

# 先确认数据集在（约 99 MB，不在的话看 data.py 文档字符串）
python3 data.py

# 主实验：8 条路线 × 延迟 + 真实 top-1（约 10 分钟）
python3 exp_quantization.py

# 追加实验：batch 扫描，验证 INT8 的交叉点（约 20 分钟）
python3 exp_quant_batch.py
```

建议自己改的两个地方：

1. `exp_quant_batch.py` 里的 `BATCHES`，加上 128 看 INT8 优势会不会继续扩大。
2. `quant_routes.py` 里 `_MODELOPT_CFGS`，换成 `INT8_SMOOTHQUANT_CFG`
   （SmoothQuant，对 transformer 更友好）或 `INT4_BLOCKWISE_WEIGHT_ONLY_CFG`
   看 INT4 在小 CNN 上是什么表现（预期是负收益，理由见 8.8）。

---

← [第 7 章：附录](07_appendix.md) | [第 9 章：动态 shape](09_dynamic_shapes.md) →
