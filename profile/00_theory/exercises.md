# 第 00 章 练习

## 练习 1（计算）：Amdahl 定律

慢流水线（`PipelineConfig.slow()`）的 p50 是 11.4 ms。已知关掉 `eager_small_ops` 后降到 7.5 ms。

1. 后处理里的小算子大约占总耗时的多少？
2. 如果把这部分优化到 0，整体最多能快多少倍？

<details><summary>参考思路</summary>

1. 节省了 11.4 − 7.5 = 3.9 ms。由于优化后后处理本身还要花一点时间，实际占比略高于 3.9 / 11.4 ≈ 34%。
2. 假设占比 p ≈ 0.34、加速比 s → ∞，Speedup = 1 / (1 − 0.34) ≈ 1.52x。与实测的 1.51x 吻合。
</details>

## 练习 2（动手）：测出你自己的拐点

修改 [02_roofline.py](02_roofline.py)，加入下列算子并标到 Roofline 上：

- `torch.nn.functional.gelu`（fp16，4096×8192）
- `torch.bmm`，形状 (64, 256, 64) × (64, 64, 256)，fp16（相当于 attention 的 QK^T）
- `F.scaled_dot_product_attention`，B=8、H=8、T=1024、d=64，fp16

问：哪些是 memory-bound？SDPA 的 AI 怎么算？

<details><summary>参考思路</summary>

- gelu：FLOPs 大约是每元素 ~10，bytes = 4/元素（fp16 读 + 写），AI ≈ 2.5，memory-bound。
- bmm：FLOPs = 2·64·256·64·256 ≈ 0.54 G，bytes = 2·64·(256·64·2 + 256·256) ≈ 12.6 MB，AI ≈ 43，在 FP16 下是 memory-bound（拐点约 690）。
- SDPA（Flash 实现）：FLOPs ≈ 4·B·H·T²·d ≈ 17.2 G，bytes ≈ 读 Q/K/V + 写 O = 4·B·H·T·d·2 ≈ 67 MB，AI ≈ 256。T 越长 AI 越高，这正是 FlashAttention 的意义：不把 T×T 的中间矩阵写回 DRAM，从而把 AI 提上来。
</details>

## 练习 3（思考）：为什么 GEMM 8192 比 4096 慢？

实测 FP16 GEMM 4096 能达到 131 TFLOPS，8192 却只有 105 TFLOPS。请给出两个可能的原因，并说明怎么验证。

<details><summary>参考思路</summary>

1. **功耗墙**：8192 单次就要 10 ms，30 次迭代就是持续 0.3 秒以上的满载，会触发功耗墙（0.4 坑 B）。验证方法：在 8192 的每次迭代之间插入 `time.sleep(0.05)` 让 GPU 冷却，看单次耗时是否下降；或者用 jtop 观察功耗曲线。
2. **wave quantization / tile 选择**：cuBLAS 为不同尺寸选择不同的 kernel 和 tile 大小，产生的 block 数除以 20 个 SM 不一定整除。验证方法：用 nsys 看两个尺寸各用了哪个 kernel（第 04 章）；用 ncu 看 launch 配置和 SM 利用率（第 05 章）。

实测验证：`benchmark` 的结果里，8192 的 min 和 p50 几乎相同（107.9 vs 106.7 TFLOPS），4096 的 min 和 p50 差别很大（159 vs 130）。这说明 8192 是"稳定地慢"，更像是 kernel 本身或者持续功耗的问题。
</details>

## 练习 4（动手）：MFU 与 batch

在 [03_mfu.py](03_mfu.py) 中，ToyVLA fp16 从 batch=32 到 batch=128 时，MFU 反而从 12.7% 降到 10.9%。请推测原因，并设计实验验证。

<details><summary>参考思路</summary>

可能的原因：(1) batch=128 时激活值的总量远大于 L2，访存密集算子（BN、LayerNorm、逐元素）的比重上升；(2) 持续满载 41 ms 触发了功耗墙；(3) 卷积的算法选择发生了变化。
验证方法：用 `torch.profiler` 统计 CPU 侧各类算子的调用次数和 shape（第 03 章），用 nsys 统计 batch=32 和 128 时各 kernel 的耗时占比（第 04 章），看哪一类 kernel 的占比变大了。
</details>

## 练习 5（思考）：利用率陷阱

同事说："这个模型的 GPU 利用率已经 97% 了，GPU 已经跑满，没有优化空间了。" 请用本章的知识反驳，并给出接下来应该看的 2 个指标。

<details><summary>参考思路</summary>

GPU Util 只表示"采样周期内有 kernel 在跑的时间比例"，即使每个 kernel 只用了 1 个 SM，它也可能显示 100%。实测 ToyVLA fp16 batch=128 时 GPU busy 97%，MFU 只有 10.9%。
接下来应该看：(1) MFU（算出实际的有效算力）；(2) 各 kernel 在 Roofline 上的位置或 ncu 的 SOL（Speed Of Light）百分比，看它们离屋顶还有多远；条件允许的话，再看 SM Active 和 Tensor Active。
</details>
