# 第 01 章 练习

## 练习 1（动手）：给自己的负载做体检

把 [03_load_patterns.py](03_load_patterns.py) 里的某个阶段换成慢流水线的完整版 `SlowPipeline(PipelineConfig.slow())`，运行后回答：

1. GPU load、最忙 CPU 核、GPU 功耗分别是多少？
2. 按照 1.7 的判读流程，它属于哪一类？下一步该用什么工具？

<details><summary>参考思路</summary>

慢流水线同时有 Python 预处理（CPU）、后处理小算子（launch）、`.item()` 同步，GPU load 应该偏低（< 50%），最忙的 CPU 核接近 100%，GPU 功耗很低。按判读流程属于"CPU-bound 或 Launch-bound"，下一步用第 02 章的 VizTracer/py-spy 看 Python 热点，再用第 04 章的 nsys 看 kernel 之间的空隙。
</details>

## 练习 2（思考）：为什么 memory-bound 阶段 CPU 核也是 100%？

1.5 节里 memory-bound 阶段最忙的 CPU 核是 100%，但 `y.copy_(x)` 只是一个 kernel，CPU 侧没有什么计算。请解释原因，并说明怎么验证。

<details><summary>参考思路</summary>

每次 copy 在 GPU 上大约要 2 ms，CPU 下发一次只要几微秒，所以 stream 队列很快就满了。队列满了以后，下一次 `cudaLaunchKernel` 必须等待空位，CUDA 默认的等待方式是自旋（spin），这会占满一个核。
验证方法：(1) 用 py-spy dump 查看线程栈，应该停在 copy_ 的调用里（第 02 章）；(2) 用 nsys 看 CUDA API 行，`cudaLaunchKernel` 的耗时会从微秒级变成毫秒级（第 04 章）；(3) 每次 copy 之后调用 `torch.cuda.synchronize()`，CPU 的负载会下降（同步时是阻塞等待，而不是自旋）。
</details>

## 练习 3（动手）：测量采样延迟

写一个脚本：t=0 时启动 `JtopSampler(0.05)`，t=2 s 开始做 GEMM，t=4 s 停止。从记录里找出 `gpu_load` 和 `p_gpu_w` 第一次超过 50 的时间点，计算 jtop 的延迟。对 `NvmlSampler` 做同样的测试，比较两者。

<details><summary>参考思路</summary>

本机实测 jtop 的延迟约为 1 s（1.5 节的图）。NVML 的 utilization 本身就是滑动窗口平均（窗口约 1 s），所以也会有延迟。结论：两种工具都不适合观察亚秒级的事件。
</details>

## 练习 4（思考）：Jetson 和独显平台的监控差异

把同一个模型部署到 x86 + RTX 显卡的服务器上，1.7 节的判读流程有哪些地方可以做得更好？

<details><summary>参考思路</summary>

独显平台上：nvidia-smi 的 `mem%`、频率、`clocks_event_reasons`（可以直接看到是否因功耗墙或温度降频）都可用；有 PCIe 吞吐指标，可以判断 H2D 拷贝是不是瓶颈；数据中心卡上可以用 DCGM 的 `SM_ACTIVE` / `SM_OCCUPANCY` / `TENSOR_ACTIVE`，直接得到空间维度的利用率，从而区分 memory-bound 和 compute-bound，不需要用功耗间接推断。
另一方面，独显平台多了一个瓶颈来源：PCIe 拷贝（Jetson 是统一内存，没有这个问题）。
</details>
