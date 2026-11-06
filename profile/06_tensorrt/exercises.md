# 第 06 章 练习

## 练习 1（动手）：batch 对 ToyVLA 的影响

把 [01_export_onnx.py](01_export_onnx.py) 中 ToyVLA 的 batch 改成 4 和 16，重新导出、构建 fp16 engine，用 [04_three_steps.py](04_three_steps.py) 的第 2 步方法测量每次推理的 kernel 间隙占比，以及 `--useCudaGraph` 的收益。预测：batch 增大后间隙占比和 CUDA Graph 的收益会怎样变化？

<details><summary>参考思路</summary>

batch=1 时 70 个 kernel 平均每个约 5 μs，间隙占 27%。batch 增大后每个 kernel 的计算量随之增大，而 launch 开销基本不变，所以间隙占比应该下降，CUDA Graph 的收益也会变小；同时 Tensor Active（第 04 章 4.8 中 batch=1 只有约 2%）会上升。吞吐（样本/秒）会明显提高，但单个请求的延迟也会增加。这正是"延迟 vs 吞吐"的取舍。（预期结果，未实测。）
</details>

## 练习 2（动手）：能不能让 stem 卷积变快？

6.4 第 3 步发现 TinyCNN 的 stem 卷积（3 通道、7×7、stride 2）只发挥了 FP16 峰值算力的约 17%。尝试以下改法之一，重新导出并走一遍三步法，比较 stem 的耗时和整个模型的 GPU Compute Time：
- (a) 用 space-to-depth：先把输入 reshape 成 `(B, 12, 112, 112)`（2×2 的空间块折叠到通道），再用 4×4 的卷积；
- (b) 输入在预处理时就 pad 成 4 通道（或 8 通道）的 fp16 NHWC，省掉 `__myl_CastConc`；
- (c) 只对 stem 用 int8（`--int8 --fp16` + `--layerPrecisions`）。

<details><summary>参考思路</summary>

(a) 改变了模型结构（需要重新训练或转换权重），但通道数变大后，Tensor Core 能用上更高效的通用卷积 kernel；(b) 不改变计算，只省掉一次格式/精度转换（6.3 中占 5.6%）；(c) int8 的 stem 实测只要 0.0617 ms（fp16 是 0.1401 ms），但需要校准，否则精度不可用。判断是否值得改，先看它占总时间的比例（27.6%）：即使 stem 降到 0，整体也最多快约 1.4 倍（Amdahl 定律，第 00 章）。（预期结果，未实测。）
</details>

## 练习 3（思考）：为什么第 1 步看不到 ToyVLA 的瓶颈？

ToyVLA 的逐层 profile 中最慢的层只占 4.3%，但 nsys 显示每次推理有 27% 的时间是 kernel 之间的间隙。为什么 TRT profiler 看不到这部分时间？逐层时间之和（0.602 ms）比真实的 GPU Compute Time（0.455 ms）还大，又该怎么解释？

<details><summary>参考思路</summary>

`--dumpProfile` 在每层前后插入 event 和同步，层与层之间的等待被同步"吸收"掉了，不属于任何一层；同时插入同步本身也让每层的时间变长（kernel 启动后才开始计时、结束后要等 event），所以总和反而比真实值大。逐层 profile 适合回答"哪层的 kernel 慢"，不适合回答"时间整体花在哪里"，后者要用 nsys 的时间线（第 04 章 4.3 的 GPU busy / 间隙分析）。
</details>

## 练习 4（动手）：严格 fp32

用 `trtexec --noTF32` 重新构建 ToyVLA 的 fp32 engine，比较 GPU Compute Time、Engine Inspector 中的 kernel 名（是否还有 `tf32` 字样），以及与 PyTorch fp32 的误差（6.5 中 TF32 版本是 4.5e-4）。

<details><summary>参考思路</summary>

关闭 TF32 后，GEMM/卷积只能用 CUDA core 的 FFMA，第 00 章实测 FP32 峰值 6.39 TFLOPS，而 TF32 约 65 TFLOPS，所以计算密集的层会明显变慢；kernel 名中的 `tf32` 应该消失；误差会降到 1e-5 量级或更低。（预期结果，未实测。）
</details>

## 练习 5（动手）：把 TensorRT engine 接进慢流水线

在 `common/pipeline.py` 的 `infer()` 中，把 `self.model(x, self.tokens)` 换成 [05_compare.py](05_compare.py) 中的 `TRTModule`（fp16 + CUDA Graph），用 `common/selftest.py` 的方法测量快版流水线的端到端延迟，和第 04 章的 nsys 结果对比：GPU busy 会变高还是变低？

<details><summary>参考思路</summary>

推理部分从约 3 ms（PyTorch fp16 eager）降到约 0.35 ms，端到端延迟会明显下降，但 GPU busy 很可能反而**降低**：GPU 干活更快了，CPU 侧的预处理/后处理没变，所以 GPU 空闲的比例更大。这说明优化完推理之后，下一个瓶颈转移到了 CPU 侧，是第 08 章综合实战的出发点。（预期结果，未实测。）
</details>
