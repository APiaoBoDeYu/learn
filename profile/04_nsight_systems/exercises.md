# 第 04 章 练习

## 练习 1（动手）：给 TinyCNN 找布局转换

用 nsys 分别采集 TinyCNN 在 fp32 和 fp16 下的推理（batch=8），用 `cuda_gpu_kern_sum:base` 统计 `nchwToNhwcKernel` 这类布局转换 kernel 的占比。再把模型和输入改成 `memory_format=torch.channels_last`，看看 fp32 下的转换能不能消除。

<details><summary>参考思路</summary>

```python
model = TinyCNN().cuda().to(memory_format=torch.channels_last)
x = x.to(memory_format=torch.channels_last)
```
cuDNN 的 TF32/FP16 卷积 kernel 偏好 NHWC 格式。输入是 NCHW 时，cuDNN 要插入 `nchwToNhwcKernel` / `nhwcToNchwKernel` 做转换；改成 channels_last 后，转换 kernel 应该会大幅减少，可以在 `cuda_gpu_kern_sum` 中对比实例数。这和 TensorRT 中的 reformat 层是同一类问题。
</details>

## 练习 2（动手）：量化 Launch-bound

对 `ManySmallOps(n)` 取 n = 10、50、200、1000，用 nsys 和 [02_analyze_sqlite.py](02_analyze_sqlite.py) 的方法计算 GPU busy 比例和 kernel 间隙的中位数，并画出"n vs GPU busy"的曲线。

<details><summary>参考思路</summary>

（预期结果，未在本机逐一实测）每个 kernel 在 GPU 上只要约 1.2 μs（4.3 节：后处理的 kernel 100% 短于 5 μs），而 CPU 发起一次要 4~5 μs（nsys 下约 7 μs），所以 n 增大后，GPU busy 应该稳定在一个很低的比例，kernel 间隙的中位数约等于单次 launch 的耗时。这说明问题不在于 kernel 的数量，而在于"每个 kernel 的计算量远小于 launch 开销"，只有融合（compile）或 CUDA Graph 能解决。
</details>

## 练习 3（思考）：GPU 投影时间为什么比 CPU 时间长？

4.2 节中慢版 infer 阶段的 GPU 投影时间（4.27 ms）比 CPU 时间（3.58 ms）长，而快版两者几乎相等（3.10 vs 3.17）。请解释原因。

<details><summary>参考思路</summary>

慢版是 fp32，GPU 上的 GEMM 和卷积更慢（4.4 节：addmm 1.41 ms vs 0.21 ms），GPU 执行速度跟不上 CPU 发起 kernel 的速度，kernel 在队列里排队（launch→start 延迟 260 μs），所以 CPU 离开 infer 之后，GPU 还在执行它发起的工作。快版是 fp16，GPU 执行得比 CPU 发起得快，每个 kernel 一被发起就立即执行（延迟 6.5 μs），GPU 投影时间自然就等于 CPU 时间。
推论：快版要提速，必须减少 CPU 侧的开销（CUDA Graph）；慢版首先应该降低 GPU 的计算量（fp16）。
</details>

## 练习 4（动手）：验证 Thor 上的拷贝/计算重叠

4.5 节 A2 中拷贝和计算严格交替。设计一个实验来区分以下两种假设：(a) 拷贝和 kernel 在硬件上互斥；(b) 只是和"占满所有 SM"的 kernel 互斥。提示：把 A2 的 GEMM 换成 B2 中只用 2 个 block 的 Triton kernel，再观察时间线。

<details><summary>参考思路</summary>

（以下是新容器中用 GPU metrics 得到的实测结论，见 4.8 B/C。）答案是 (b) 的一种更具体的形式：**≥256 KB 的拷贝本身就是由 SM 执行的**，拷贝期间 Copy Engine 活动为 0、SMs Active 约 99.8%。所以它和占满 SM 的 GEMM 只能交替执行；换成只用 2 个 block 的 kernel 后，空闲的 SM 可以用来执行拷贝，预期能看到重叠（这一步未实测，留作练习）。小于约 256 KB 的拷贝走 Copy Engine，可以和 GEMM 真正重叠，但带宽只有约 14~20 GB/s。用 `nsys export` + SQL（4.3）比较 memcpy 和 kernel 的 start/end，再配合 `GPU_METRICS` 表中的 `Async Copy Engine Active` 和 `SMs Active` 即可判断。
</details>

## 练习 5（动手）：给慢流水线的后处理套上 CUDA Graph

慢流水线的后处理里有 `.item()`，不能直接 capture。先把它改造成无同步的版本（参考第 03 章练习 1），再用 CUDA Graph capture 后处理，测量延迟的变化，并用 nsys 确认 launch 次数从约 378 次降到 1 次。

<details><summary>参考思路</summary>

改造要点：(1) 去掉循环中的 `.item()`，越界标志留在 GPU 上；(2) 输入 `raw`、`prev_action` 都要用静态缓冲区，每步 `copy_` 进去；(3) 输出也是静态缓冲区，最后统一 `.cpu()`。预期效果（根据 4.3 节的数据推算，未实测）：后处理在 GPU 上的 kernel 时间约 0.5 ms（GPU busy 0.488 ms），capture 后 CPU 侧只剩 1 次 `cudaGraphLaunch`，后处理的耗时应该从约 6 ms 降到接近 0.5 ms。
</details>
