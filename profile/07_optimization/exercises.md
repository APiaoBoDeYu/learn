# 第 07 章 练习

## 练习 1（动手）：手写 GPU 预处理 kernel

7.1 中 A4（GPU 上 `F.interpolate` + 归一化）一共发起了多个 kernel（类型转换、插值、除法、减法、除法）。用 Triton 或第 05 章的 nvcc + ctypes 方式，写一个融合 kernel：输入 480×640×3 的 uint8 HWC，直接输出 1×3×224×224 的归一化 float（双线性插值 + /255 + 减均值 + 除方差 + HWC→CHW 一步完成）。和 A4 比较耗时，并用 nsys 数一下 kernel 数量。

<details><summary>参考思路</summary>

每个输出像素读 4 个输入像素（双线性），写 1 个 float。输出只有 0.6 MB，所以这个 kernel 的耗时主要由 launch 和输入读取决定，预期能比 A4 的多个 kernel 更快（未实测）。插值的坐标公式要和 `cv2.INTER_LINEAR`（`align_corners=False`、像素中心对齐）一致，否则会出现 7.1 中 1e-2 量级的差异。
</details>

## 练习 2（动手）：权重量化对 decode 的影响

7.5 的 decode 是 Memory-bound：读一遍 204 MB 的 fp16 权重至少要 0.848 ms。把所有 Linear 的权重存成 int8（每个输出通道一个 scale），在 forward 中先反量化再做 matmul，测量 decode 一步的耗时。为什么这种"先反量化"的写法可能并不会变快？要怎么改？

<details><summary>参考思路</summary>

如果先把 int8 权重反量化成 fp16 的完整矩阵，再调用 fp16 GEMV，读的字节数反而更多（读 int8 + 写 fp16 + 读 fp16）。真正的收益来自**在 kernel 内部反量化**（weight-only quantization kernel，例如 TensorRT-LLM 的 int8/int4 weight-only GEMM，或 torchao），这样从 DRAM 读入的只有 int8 权重，字节数减半，预期 decode 接近 2 倍。（预期结果，未实测。）
</details>

## 练习 3（思考）：为什么 ManySmallOps 的 CUDA Graph 只有 6.38x，而 compile 有 33.8x？

<details><summary>参考思路</summary>

CUDA Graph 消除的是 CPU 侧每次 launch 的开销（约 5 μs/次），但 300 个 kernel 仍然要在 GPU 上一个接一个执行，每个 kernel 本身的启动和执行也需要约 0.9 μs（0.261 ms / 300）。compile 把 300 个算子融合成 1 个 kernel：既没有 launch 开销，也没有 kernel 之间的间隔，中间结果也不用写回显存。所以"减少 kernel 数量"永远比"让 kernel 启动得更快"更有效，前提是能融合。
</details>

## 练习 4（动手）：decode 的 batch 与延迟预算

假设机器人控制回路要求每次决策（生成 16 个动作 token）的延迟 ≤ 50 ms。用 7.5 C 的数据估算：batch 最大能开到多少？如果改成 7.5 D 的 CUDA Graph 版本呢？

<details><summary>参考思路</summary>

batch=16 时每步 4.357 ms，16 步约 70 ms，超出预算；batch=4 时每步 2.370 ms，16 步约 38 ms，满足要求。所以 batch 最多约为 4~8（batch=8 未测，可以插值后实测验证）。CUDA Graph 在 batch=1 时只快了 1.11x，对结论影响不大。注意这里没有算 prefill 和预处理的时间，真实系统需要整条流水线一起测（第 08 章）。
</details>

## 练习 5（动手）：channels_last + torch.compile

在 7.3 B 的基础上，再对 fp16 channels_last 的 TinyCNN 使用 `torch.compile`，看看还能快多少；再和第 06 章 TensorRT fp16 的 TinyCNN（0.474 ms，不含 CUDA Graph）对比。

<details><summary>参考思路</summary>

第 06 章 6.5 中 compile fp16（NCHW 输入）是 0.852 ms；inductor 通常会自己选择内存布局，所以 channels_last + compile 和单独 compile 的差别可能不大（未实测）。和 TensorRT 的差距主要来自 TensorRT 对卷积 + BN + ReLU + Add 的融合和 tactic 选择（第 06 章 6.3）。
</details>
