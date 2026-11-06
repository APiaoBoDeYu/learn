# 第 06 章 推理引擎：TensorRT 性能分析

> 📎 对应笔记：[doc/TensorRT_副本.xmind](../doc/TensorRT_副本.xmind) →「性能分析」：内置 Profiler、Nsight Systems、trtexec、区别联系、"为什么 TRT Profiler 和 Nsight 要配合使用"、三步法
>
> 一句话分工（📎 笔记·区别联系）：**TRT Profiler 回答"哪层慢"，Nsight Systems 回答"为什么慢"，Nsight Compute 回答"这个 kernel 为什么慢"**。TRT Profiler 懂模型语义（层名）但看不到硬件；Nsight 看得到硬件，但只认 kernel 名。本章用 NVTX 把两边对上，完整走一遍三步法。
>
> 环境：TensorRT 10.14.1（Python 包 + `/usr/bin/trtexec`）、nsys 2025.3.2、ncu 2025.3.1，新容器（privileged）。所有数字均为本机实测。

## 学习目标

1. 会用 trtexec 构建 FP32/FP16/INT8 engine，读懂 Latency / GPU Compute Time / Enqueue Time，知道 `--useCudaGraph`、`--noDataTransfers`、`--useSpinWait` 各自影响什么；
2. 会用 TensorRT 内置 profiler 和 Engine Inspector 找到"哪层慢"、每层用了哪个 kernel、做了哪些融合；
3. 完整走通三步法：TRT Profiler → nsys（NVTX 把层和 kernel 对应起来）→ ncu（只采某一层的 kernel）；
4. 对比同一个模型在 PyTorch eager / torch.compile / TensorRT 下的延迟和精度。

## 目录

- [6.1 导出 ONNX](#61-导出-onnx)
- [6.2 trtexec 基准测试](#62-trtexec-基准测试)
- [6.3 内置 Profiler 与 Engine Inspector（第 1 步：哪层慢）](#63-内置-profiler-与-engine-inspector第-1-步哪层慢)
- [6.4 三步法：TRT Profiler → nsys → ncu](#64-三步法trt-profiler--nsys--ncu)
- [6.5 PyTorch vs TensorRT](#65-pytorch-vs-tensorrt)
- [6.6 精度、Plugin、torch2trt、DLA](#66-精度plugintorch2trtdla)
- [常见坑](#常见坑) · [练习](exercises.md) · [小结](#小结)

| 文件 | 内容 |
|---|---|
| [01_export_onnx.py](01_export_onnx.py) | 导出 TinyCNN（batch=8）、ToyVLA（batch=1）的 ONNX，保存参考输入/输出 |
| [02_trtexec.py](02_trtexec.py) | 构建 FP32/FP16/INT8 engine；4 种推理选项的对比 |
| [03_layer_profile.py](03_layer_profile.py) | `--dumpProfile --separateProfileRun --exportProfile --exportLayerInfo`：逐层耗时、kernel、格式、融合 |
| [04_three_steps.py](04_three_steps.py) | 三步法：逐层 profile → nsys（层 → kernel、kernel 间隙、CUDA Graph）→ ncu（只采某一层） |
| [05_compare.py](05_compare.py) | PyTorch eager / compile / TensorRT（±CUDA Graph）的延迟与精度 |

---

## 6.1 导出 ONNX

[01_export_onnx.py](01_export_onnx.py)，日志 [outputs/01_export_onnx.log](outputs/01_export_onnx.log)。

```python
torch.backends.mha.set_fastpath_enabled(False)      # 否则 TransformerEncoderLayer 走融合 fastpath，导出器不认识
torch.onnx.export(m, inputs, "x.onnx", input_names=[...], output_names=["out"], opset_version=17, dynamo=False)
```

| 模型 | ONNX | 节点数 | 主要算子 |
|---|---|---|---|
| TinyCNN（8×3×224×224） | 21.7 MB | 37 | Conv×12、Relu×9、Identity×8（BN 已被导出器折叠进 Conv）、Add×4 |
| ToyVLA（1×3×224×224 + 1×32 tokens） | 100.2 MB | 642 | Constant×161、Identity×74、Reshape×67、Add×51、Transpose×49、MatMul×33、LayerNormalization×13 |

- 新镜像（v1.3.0）中没有 onnxscript，torch 2.9 默认的 dynamo 导出器不可用，所以用 `dynamo=False`（TorchScript 导出器）。
- ToyVLA 的 642 个节点中有大量 Reshape/Transpose/Constant，这些在 TensorRT 中几乎都会被消除或融合（6.3）。

## 6.2 trtexec 基准测试

[02_trtexec.py](02_trtexec.py)，日志 [outputs/02_trtexec.log](outputs/02_trtexec.log)（每次 trtexec 的完整输出在 `outputs/02_build_*.log`、`02_run_*.log`）。

```bash
trtexec --onnx=x.onnx --saveEngine=x_fp16.engine --profilingVerbosity=detailed --fp16 --skipInference   # 构建
trtexec --loadEngine=x_fp16.engine --duration=3 --warmUp=500 [--useCudaGraph | --noDataTransfers | --useSpinWait]
```

> 📎 笔记中的 `--workspace=1024` 在 TensorRT 10 中已经移除，改用 `--memPoolSize=workspace:1024M`。

### 构建

| 模型 | 精度 | 构建时间 | engine 大小 |
|---|---|---|---|
| TinyCNN | fp32 / fp16 / int8 | 12.4 / 15.5 / 21.3 s | 23.7 / 12.2 / 6.7 MB |
| ToyVLA | fp32 / fp16 / int8 | 15.5 / 27.0 / 34.9 s | 101.3 / 51.3 / 42.8 MB |

- **fp32 默认允许 TF32**：TensorRT 在 fp32 模式下会用 TF32 Tensor Core 算卷积和矩阵乘（6.3 中能看到 `tf32` 字样的 kernel），严格 fp32 需要加 `--noTF32`。
- **int8 没有提供校准数据**：trtexec 会给所有张量设置一个占位的动态范围，**只能用来看性能，精度没有意义**（6.5 中 ToyVLA 的误差是 0.83）。真正部署需要校准器，或者用 Q/DQ 量化过的 ONNX。`--int8 --fp16` 表示不能用 int8 的层退回 fp16。

### 推理选项（单位 ms，取中位数；GPU = GPU Compute Time，Lat = H2D + GPU + D2H）

| 模型 / 精度 | 默认 GPU / Lat / Enqueue | `--useCudaGraph` GPU / Enqueue | `--noDataTransfers` GPU | `--useSpinWait` GPU / Lat |
|---|---|---|---|---|
| TinyCNN fp32 | 0.920 / 1.062 / 0.082 | 0.863 / 0.004 | 0.886 | 0.909 / 0.958 |
| TinyCNN fp16 | 0.481 / 0.557 / 0.069 | 0.435 / 0.003 | 0.436 | 0.463 / 0.498 |
| TinyCNN int8 | 0.350 / 0.400 / 0.084 | 0.312 / 0.003 | 0.345 | 0.369 / 0.419 |
| ToyVLA fp32 | 1.038 / 1.062 / 0.240 | 0.958 / 0.003 | 1.044 | 1.065 / 1.100 |
| ToyVLA fp16 | 0.458 / 0.476 / 0.235 | **0.334** / 0.003 | 0.457 | 0.456 / 0.478 |
| ToyVLA int8 | 0.433 / 0.450 / 0.247 | 0.323 / 0.003 | 0.429 | 0.454 / 0.470 |

读法：
- **GPU Compute Time** 是一次 enqueue 从第一个 kernel 开始到最后一个 kernel 结束的时间，**包含 kernel 之间的空隙**。所以 `--useCudaGraph` 不只是降低了 Enqueue Time（0.235 → 0.003 ms），连 GPU Compute Time 也下降了：ToyVLA fp16 从 0.458 降到 0.334 ms（-27%），说明原本有约四分之一的 GPU 时间花在等 CPU 发 kernel 上（6.4 第 2 步会直接量出来）。TinyCNN 的 kernel 少而大，只下降约 10%。
- **H2D/D2H**：Thor 是统一内存，TinyCNN 的 4.8 MB 输入 H2D 只要约 0.05 ms，所以 `--noDataTransfers` 和默认的 GPU 时间几乎一样，Latency 只比 GPU 时间多 0.02~0.14 ms。在独显平台上，这一项通常大得多。
- **`--useSpinWait`** 在这里没有稳定的收益（± 几个百分点）：它节省的是 CPU 被唤醒的延迟，只在 Latency 很短、对尾延迟敏感的场景下才值得用，代价是占满一个 CPU 核。
- **INT8**：TinyCNN 比 fp16 快 1.37 倍，ToyVLA 几乎没有收益（0.433 vs 0.458 ms）。ToyVLA 在 batch=1 时受 launch 间隙和小 kernel 限制（下一节），计算本身占比不高。

## 6.3 内置 Profiler 与 Engine Inspector（第 1 步：哪层慢）

[03_layer_profile.py](03_layer_profile.py)，日志 [outputs/03_layer_profile.log](outputs/03_layer_profile.log)。

```bash
trtexec --loadEngine=x.engine --dumpProfile --separateProfileRun \
        --exportProfile=profile.json --exportLayerInfo=layers.json
```

- `--dumpProfile`：逐层计时，原理是在每层前后插入同步和 event，所以**有开销**：逐层时间之和比不挂 profiler 时的 GPU Compute Time 多 7~34%（TinyCNN fp32 0.943 vs 0.881 ms，+7%；ToyVLA fp16 0.602 vs 0.455 ms，+32%；ToyVLA int8 0.589 vs 0.439 ms，+34%）。层越多越小，偏差越大。
- `--separateProfileRun`：性能数字和逐层数字来自两轮不同的运行，互不干扰。
- `--exportLayerInfo`（Engine Inspector）：每层的 `LayerType`、输入输出的格式和精度、`TacticName`（**就是实际调用的 kernel 名**）、`Metadata`（这个 TRT 层融合了哪些 ONNX 层）。需要构建时加 `--profilingVerbosity=detailed`，否则只有层名。也可以用 Python 的 `engine.create_engine_inspector()` 获取。

### TinyCNN fp16（30 个 TRT 层）

| avg ms | % | TRT 层 | 类型 | 输入→输出 | 融合的 ONNX 层 | kernel |
|---|---|---|---|---|---|---|
| **0.1401** | **27.6** | `/stem/stem_0/Conv_myl0_2` | correlation | Half→Half | 2（Conv + Relu） | `cutlass_tensorop_first_layer_fprop_f16_f16_f32_..._align4_filter_64x7x7x4` |
| 0.0478 | 9.4 | `/stem/stem_3/MaxPool_myl0_3` | maxpool | Half→Half | 1 | `sm50_xmma_pooling_coalescedC_NHWC_kMAX_3_False` |
| 0.0403 | 7.9 | `/layers/layers_0/conv2/Conv_myl0_5` | correlation | Half→Half | 3（Conv + Add + Relu） | `cutlass3x_sm100_tensorop_s64x128x16_conv3d_fprop_...` |
| 0.0282 | 5.6 | `__myl_CastConc_myl0_1` | kgen | **Float→Half** | 2 | Myelin 生成的 kernel |

- **TensorRT 10.14 在 Thor 上把整张图交给 Myelin 编译器**（层名的 `_myl0_` 后缀；nsys 中整张图对 TRT 来说只是一个 `{ForeignNode[...]}`）。卷积的 BN/ReLU/残差 Add 都被融合进了卷积（"融合的 ONNX 层"一列）。
- **格式转换**：没有出现旧版本中独立的 `Reformat` 层，取而代之的是 `__myl_CastConc`：把 fp32 的 NCHW 输入转成 fp16，**并把 3 个通道 pad 到 4**（输出形状是 `8×4×224×224`），以满足第一层 Tensor Core kernel 的对齐要求（kernel 名中的 `align4_filter_64x7x7x4`）。这一层占 5.6%，是第 04 章 `nchwToNhwcKernel` 和大纲中"reformat 层"在 TensorRT 中的对应物。
- **按类型汇总**：correlation（卷积）78%、kgen（Myelin 生成的融合 kernel）12%、maxpool 9%。

### ToyVLA fp16（84 个 TRT 层）

| avg ms | % | TRT 层 | kernel |
|---|---|---|---|
| 0.0257 | 4.3 | `/stem/stem_0/Conv_myl0_2` | `sm80_xmma_fprop_implicit_gemm_indexed_wo_smem_f16f16_...` |
| 0.0150 | 2.5 | `/layers/layers_3/conv2/Conv_myl0_26` | `cutlass3x_sm100_tensorop_s64x64x16implicit_gemm_...` |
| 0.0140 | 2.3 | `__myl_CastGathConcAddCastMeanSubMulMeanAddSqrtDivMul...`（融合了 5 个 ONNX 层：embedding Gather + Concat + 位置编码 Add + LayerNorm） | Myelin 生成 |
| 0.0139 | 2.3 | `/backbone/encoder/layers_0/linear1/MatMul...`（融合了 7 个 ONNX 层） | `cutlass3x_sm100_tensorop_h256x64x16gemm_f16_...` |

- **没有"最慢的那一层"**：最慢的层只占 4.3%，按类型汇总是 kgen 58%、correlation 25%、gemm 15%。📎 笔记中"找到 `MatMul_387 40ms` 这种热点层"的情形并不总会出现；对这种很平的分布，第 1 步给不出答案，只能进入第 2 步看整体（6.4）。
- 642 个 ONNX 节点被压缩成了 84 个 TRT 层、每次推理 70 个 kernel（6.4）。

### INT8 的代价

TinyCNN int8 中出现了 `__myl_MulMinMaxRounCastConc`（fp32 输入 → int8 量化）、`__myl_CastMulCastCastMulMinMaxRounCast`（重新量化）这类 kgen 层，kgen 的占比从 fp16 的 12% 增加到 24%；另外出现了两个 int8 pooling 层（`__mye2643_pooling`、`__mye2715_pooling`，合计 23%），而原模型只有一个 MaxPool（原因未深究）。INT8 能让卷积本身快约 2 倍（stem 卷积 0.1401 → 0.0617 ms），但量化/反量化的额外 kernel 会吃掉一部分收益，整体是 1.37 倍（6.2）。

## 6.4 三步法：TRT Profiler → nsys → ncu

[04_three_steps.py](04_three_steps.py)，日志 [outputs/04_three_steps.log](outputs/04_three_steps.log)。

### 第 2 步的准备：TensorRT 发出的 NVTX

```bash
nsys profile -t cuda,nvtx -o x trtexec --loadEngine=x.engine --iterations=50 --duration=0 --warmUp=0
```

> ⚠️ 大纲和笔记中的 `nsys profile -t cuda,nvtx,tensorrt` 在 nsys 2025.3 中会报 `Illegal --trace argument 'tensorrt'`，这个选项已经不存在了。TensorRT 自己会发 NVTX，`-t nvtx` 就能收到。

| NVTX domain | range | 说明 |
|---|---|---|
| `TensorRT` | `ExecutionContext::enqueueV3`、`{ForeignNode[image_cast...out_castOut]}` | 整张 Myelin 图对 TensorRT 来说只是一个 ForeignNode 层 |
| `myelin-exec` | `/stem/stem_0/Conv_myl0_2`、`__myl_CastConc_myl0_1` …… | Myelin 内部每一层一个 push/pop range，名字和第 1 步的逐层 profile 完全一致（需要 `--profilingVerbosity=detailed`） |

**层 → kernel 的对应方法**：`myelin-exec` 的 range（CPU 时间）内发起的 `cudaLaunchKernel` / `cuLaunchKernel` → `correlationId` → GPU 上的 kernel（SQL 实现见 `step2()`，和第 04 章 4.4 节"每个 PyTorch 算子的 GPU 时间"是同一个方法）。

### TinyCNN fp16：瓶颈在 kernel 本身

**第 1 步**：stem 卷积 0.1401 ms，占 27.6%。

**第 2 步**（nsys）：

| | 每次推理的 kernel | GPU 跨度 | kernel 执行（并集） | 间隙 | CPU 侧 |
|---|---|---|---|---|---|
| 默认 | 16 | 480.5 us | 449.2 us（93%） | 31.3 us | enqueueV3 133.8 us |
| `--useCudaGraph` | 16 | 436.9 us | 427.7 us（98%） | 9.1 us | cudaGraphLaunch 14.7 us |

| TRT 层 | TRT profiler | nsys 测得的 GPU 时间 | kernel |
|---|---|---|---|
| `/stem/stem_0/Conv_myl0_2` | 0.1401 ms | 0.1461 ms | 1 个：`cutlass_tensorop_first_layer_fprop_f16_f16_f32_void_f16_ndhwc...` |
| `/stem/stem_3/MaxPool_myl0_3` | 0.0478 | 0.0458 | `sm50_xmma_pooling_coalescedC_NHWC_kMAX_3_False` |
| `/layers/layers_0/conv2/Conv_myl0_5` | 0.0403 | 0.0382 | `cutlass3x_sm100_tensorop_s64x128x16_conv3d_fprop_weight_stationary...` |
| `/layers/layers_2/conv2/Conv_myl0_19` | 0.0245 | 0.0205 | `cutlass3x_sm100_tensorop_s256x128x16implicit_gemm_fprop_f16...` |

GPU 时间的 93% 都在 kernel 上，间隙只占 7%，CUDA Graph 也只快了 1.10 倍。对照📎 笔记中"为什么慢"的四种情况：kernel 不碎、没有 GPU 空转、没有独立的 reformat 层 → 剩下的是"kernel 本身慢"，进入第 3 步。

**第 3 步**（ncu，只采两层的 kernel）：

```bash
ncu --nvtx --nvtx-include "regex:myelin-exec@^.stem.stem_0.Conv_myl0_2$/" \
           --nvtx-include "regex:myelin-exec@^.layers.layers_2.conv2.Conv_myl0_19$/" \
    --section SpeedOfLight --section Occupancy ... trtexec --loadEngine=tinycnn_b8_fp16.engine --iterations=2 --duration=0 --warmUp=0
```

> ⚠️ 层名本身含有 `/`，和 NVTX 过滤语法中表示嵌套的 `/` 冲突：`"myelin-exec@/stem/stem_0/Conv_myl0_2/"` 匹配不到任何 kernel。改用正则形式 `regex:<domain>@<正则>/`，把层名中的 `/` 换成正则的 `.`。

| 层 | kernel | ncu 时间 | SM% | tensor pipe | tc pipe | DRAM%（换算） | grid | Occ% | regs |
|---|---|---|---|---|---|---|---|---|---|
| stem 卷积 | `cutlass_tensorop_first_layer_fprop_f16...` | 142.8 us | 71.4 | **75.1** | **0** | 47 | 112 | 15.3 | 189 |
| layers_2.conv2 | `cutlass3x_sm100_tensorop_s256x128x16...` | 35.6 us | 24.1 | 30.0 | 31.2 | 44 | **28** | 14.9 | 191 |

- **stem 卷积**：真实计算量 2×8×64×112×112×3×49 = 3.78 GFLOP，142.8 us 合 **26.4 TFLOPS，只有 FP16 Tensor Core 实测峰值（155.6）的约 17%**，但 tensor pipe 已经 75% 繁忙。原因是：(1) 输入通道只有 3（被 pad 到 4），每次 MMA 的 K 维度很小，算不满；(2) 这个"first layer"专用 kernel 走的是旧的 mma 指令路径（tcgen05 的 `tc pipe` 为 0，参见第 05 章 5.8 K=4095 的例子）。DRAM 只用了 47%，所以它既不是带宽瓶颈，也没有用上 Blackwell 的新 Tensor Core 路径。
- ncu 对 stem 卷积的建议：Theoretical Occupancy 16.7%（受寄存器限制，189 regs，预估加速 83.3%）、Wait Stall（40.1%）、**Tail Effect**（112 个 block，每轮 20 个，最后一轮只有 12 个，预估加速 33.3%）。
- **layers_2.conv2**：tensor 和 tc pipe 都只有约 30%，SM 只有 24%。grid 只有 28 个 block；按实测 Occupancy（约 15%，与 stem 卷积相同，即每个 SM 约 1 个 block）推算，第一轮 20 个、第二轮只有 8 个 block，尾部效应比 stem 更严重（推算，未单独打印这一层的规则建议）。batch=8 时这一层的输出只有 14×14，问题规模太小。
- 可能的修复方向（**未实测**，留作练习 2）：stem 用 int8（第 1 步中 int8 stem 只要 0.0617 ms）；把 7×7 stride 2 的 stem 换成 space-to-depth / patchify 的形式让通道数变大；增大 batch 以填满后面的小层。这些都不是 TensorRT 的选项能解决的，需要改模型。

### ToyVLA fp16：瓶颈在 kernel 之间

**第 1 步**：分布很平（最慢的层只占 4.3%），没有明确的目标。

**第 2 步**（nsys）：

| | 每次推理的 kernel | GPU 跨度 | kernel 执行（并集） | **间隙** | CPU 侧 |
|---|---|---|---|---|---|
| 默认 | 70 | 480.2 us | 351.4 us（73%） | **128.8 us（27%）** | enqueueV3 **410.9 us** |
| `--useCudaGraph` | 70 | 344.6 us | 338.6 us（98%） | 5.9 us | cudaGraphLaunch 55.5 us |

**答案在这里**：70 个 kernel 平均每个只有约 5 us，CPU 侧的 `enqueueV3` 要 411 us，和 GPU 跨度（480 us）相当，GPU 有 27% 的时间在等 CPU 发下一个 kernel，即 **TensorRT 内部也是 Launch-bound**（第 00 章 0.5 的瓶颈分类；第 04 章快版流水线的结论）。CUDA Graph 把间隙从 128.8 us 降到 5.9 us，GPU 跨度缩短 1.39 倍。这正是📎 笔记中"情况 3：多个小 kernel，launch overhead 巨大"，而它只有在第 2 步才看得到：第 1 步的逐层 profile 把每层前后都同步了，间隙不会体现在任何一层的时间里。

这里不需要第 3 步：每个 kernel 都很短，优化单个 kernel 的收益远小于消除间隙。

## 6.5 PyTorch vs TensorRT

[05_compare.py](05_compare.py)，日志 [outputs/05_compare.log](outputs/05_compare.log)。输入已在 GPU 上，CUDA Event 计时，取 200 次的中位数；误差是与 PyTorch fp32 输出的最大绝对误差。

| 方案 | TinyCNN b=8 | | ToyVLA b=1 | |
|---|---|---|---|---|
| | GPU ms（加速比） | max\|err\| | GPU ms（加速比） | max\|err\| |
| torch eager fp32 | 2.824（1.00x） | 0 | 3.555（1.00x） | 0 |
| torch eager fp16 | 1.883（1.50x） | 7.6e-05 | 2.611（1.36x） | 6.8e-04 |
| torch.compile fp16 | 0.852（3.32x） | 8.3e-05 | 1.540（2.31x） | 6.0e-04 |
| torch.compile fp16 reduce-overhead | 0.741（3.81x） | 8.3e-05 | 0.758（4.69x） | 6.0e-04 |
| TensorRT fp32（含 TF32） | 0.929（3.04x） | 5.8e-05 | 1.041（3.41x） | 4.5e-04 |
| TensorRT fp32 + CUDA Graph | 0.868（3.25x） | | 0.901（3.94x） | |
| TensorRT fp16 | 0.474（5.96x） | 7.7e-05 | 0.479（7.42x） | 9.9e-04 |
| **TensorRT fp16 + CUDA Graph** | **0.428（6.60x）** | | **0.345（10.30x）** | |
| TensorRT int8（未校准） | 0.361（7.83x） | 4.3e-02 ❌ | 0.476（7.46x） | 8.3e-01 ❌ |
| TensorRT int8 + CUDA Graph（未校准） | 0.304（9.29x） | | 0.313（11.35x） | |

（"+ CUDA Graph" 用 `torch.cuda.CUDAGraph` 捕获 `context.execute_async_v3`，和 `trtexec --useCudaGraph` 等价，误差与不加 Graph 相同。）

- **对 ToyVLA（流水线里的模型）**：TensorRT fp16 + CUDA Graph 0.345 ms，是 eager fp32 的 10.3 倍，比 torch.compile 最好的 reduce-overhead（同样用了 CUDA Graph）快 2.2 倍。第 05 章 5.9 发现快版流水线的瓶颈是大量低效的逐元素/LayerNorm 小 kernel，TensorRT（Myelin）把它们融合成了 70 个 kernel，这正是它快的原因。
- **TF32**：TensorRT fp32 默认用 TF32，所以"TensorRT fp32"与"PyTorch fp32"并不等价（PyTorch 的 matmul 默认不用 TF32）。误差 4.5e-4 是 TF32 带来的。
- **精度**：fp16 的最大误差在 1e-3 以内（ToyVLA 输出范围 [-1, 1]）。INT8 没有校准，ToyVLA 的误差达到 0.83，完全不可用，只能看性能。
- torch.compile 在 Thor 上会提示 `Not enough SMs to use max_autotune_gemm mode`：inductor 的 GEMM 自动调优要求的 SM 数比 Thor 的 20 个多，这个模式在 Thor 上用不了。

## 6.6 精度、Plugin、torch2trt、DLA

| 主题 | 本机情况 |
|---|---|
| FP16 精度排查（Polygraphy） | 新镜像中没有 polygraphy（也没有 onnxruntime），本章直接比较 TensorRT 与 PyTorch 的输出（6.5）。逐层精度排查可以用 `trtexec --layerPrecisions=... --precisionConstraints=obey` 把可疑层固定为 fp32，或安装 polygraphy 用 `polygraphy run --trt --onnxrt --trt-outputs mark all`（未实测） |
| INT8 校准 / FP8 | 未实测：需要校准数据（`IInt8EntropyCalibrator2`）或 Q/DQ 量化的 ONNX（ModelOpt，新镜像中没有） |
| Plugin 性能 | 本章的两个模型都不需要 plugin。要点（未实测）：workspace 在 `getWorkspaceSize` 中声明、不要在 `enqueue` 中分配显存或同步，否则既慢又无法被 CUDA Graph 捕获（第 04 章 4.6 的 capture 限制） |
| torch2trt / ONNX Runtime | 新镜像中都没有（旧镜像中有 torch2trt），未实测 |
| DLA / PVA | `trt.Builder().num_DLA_cores` = **0**：Thor 没有 DLA（第 01 章已确认）。PVA 不对 TensorRT 开放 |

## 常见坑

1. **`-t tensorrt` 已经不存在**：nsys 2025.3 报 `Illegal --trace argument`，用 `-t cuda,nvtx` 即可收到 TensorRT 的 NVTX。
2. **构建时没加 `--profilingVerbosity=detailed`**：Engine Inspector 只有层名，没有 kernel/格式/融合信息；nsys 中也没有 `myelin-exec` 的逐层 range。
3. **把逐层 profile 的时间当真实时间**：每层前后插入同步，总和比真实 GPU 时间多 7~34%，而且**完全看不到层与层之间的间隙**（ToyVLA 27% 的间隙在第 1 步中不可见）。
4. **ncu 的 NVTX 过滤匹配不到层名**：层名中的 `/` 与过滤语法冲突，用 `regex:myelin-exec@^<层名，/ 换成 .>$/`。
5. **以为 fp32 engine 就是 fp32 计算**：默认是 TF32，需要 `--noTF32`。
6. **用未校准的 INT8 engine 评估精度**：trtexec 的占位动态范围只用于测性能。
7. **只看 GPU Compute Time 而忽略 launch 间隙**：GPU Compute Time 包含间隙，`--useCudaGraph` 能让它下降 27%（ToyVLA fp16），说明瓶颈不在 kernel。
8. **笔记中的 `--workspace`**：TensorRT 10 改为 `--memPoolSize=workspace:<size>`。
9. **TensorRT 的 engine 与 GPU 架构、TensorRT 版本绑定**：本章的 engine 在 sm_110 + TRT 10.14 上构建，换平台必须重新构建（`outputs/` 中的 .engine 不要拷到别的机器上用）。

## 练习

见 [exercises.md](exercises.md)。

## 小结

- **三步法实测**：
  - TinyCNN：第 1 步指向 stem 卷积（27.6%）→ 第 2 步确认间隙只有 7%、CUDA Graph 只有 1.10x，问题在 kernel 本身 → 第 3 步 ncu：tensor pipe 75% 繁忙但只发挥了约 17% 的峰值算力（3 通道、旧 mma 路径）；另一层是 grid 太小（28 个 block），20 个 SM 填不满。
  - ToyVLA：第 1 步分布很平，没有热点 → 第 2 步发现 kernel 之间有 27% 的间隙、enqueue 和 GPU 时间相当 → Launch-bound，CUDA Graph 1.39x，不需要第 3 步。
- **各工具的边界**：TRT profiler 能告诉你层名、kernel、融合关系，但看不到间隙；nsys 通过 NVTX 把层和 kernel 对应起来，看得到间隙；ncu 只在"kernel 本身是瓶颈"时才值得用。
- **收益**：ToyVLA 从 eager fp32 的 3.555 ms 到 TensorRT fp16 + CUDA Graph 的 0.345 ms（10.3 倍），比 torch.compile reduce-overhead 快 2.2 倍，误差 < 1e-3。这些手段在第 07 章做系统的整理，第 08 章用到完整的流水线上。
