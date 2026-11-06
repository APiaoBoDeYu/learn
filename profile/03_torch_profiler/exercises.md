# 第 03 章 练习

## 练习 1（动手）：消除后处理中的同步

修改 `SlowPipeline.postprocess` 中 `sync_in_loop` 的逻辑：保留"统计越界维度个数"这个功能，但每步只同步一次。用 [03_sync_detection.py](03_sync_detection.py) 验证同步次数从 16 次降到 2 次（如果预处理仍然是 pageable 拷贝），再用 `benchmark` 测量延迟的变化。

<details><summary>参考思路</summary>

```python
over = (out.abs().amax(dim=0) > 0.99 * self.action_scale)   # (7,) bool，留在 GPU 上
self._over_flags = over                                       # 和动作一起，在最后的 .cpu() 时一次取回
...
result = torch.cat([out.flatten(), over.float()]).cpu()       # 只做一次 D2H
```
关键：不在循环中调用 `.item()`，把需要的标量和最终结果一起取回。
</details>

## 练习 2（动手）：让 compile 真正生效

[05_compile_diag.py](05_compile_diag.py) 中 `post_slow` 被 dynamo 完全放弃了。试试以下两种做法，分别用 counters 查看编译出了几个图，并测量速度：
1. 设置 `torch._dynamo.config.capture_scalar_outputs = True`；
2. 把 `.item()` 的比较移到循环外，改成张量运算。

<details><summary>参考思路（本机实测）</summary>

| 做法 | 编译出的图 | p50 |
|---|---|---|
| eager `post_slow` | - | 3.47 ms |
| `capture_scalar_outputs=True` + compile | **0** | 3.47 ms（没有任何变化） |
| 改成张量运算 + compile | 1 | **0.46 ms**（eager 下 3.45 ms） |

1. `capture_scalar_outputs=True` 只能让 `.item()` 本身进入图（作为符号值），但后面的 `if ... > ...:` 依赖这个值做 Python 分支判断，仍然无法 trace，函数照样被整个放弃。**光改配置是不够的**。
2. 改成张量运算（`warn = warn + (col.abs().max() > thresh).float()`，不再有 Python 分支）后，dynamo 能编译出一张完整的图：Python 循环在 trace 时被展开，16×7 个小算子被 Inductor 融合，快了 7.5 倍。
</details>

## 练习 3（思考）：为什么 sync 耗时是 0？

3.2 节中慢流水线开着 profiler 时，16 次同步一共只耗时 0.09 ms，而第 02 章 line_profiler 测得每次 `.item()` 约 0.2 ms。请解释两者为什么不一致，哪个更接近真实情况？

<details><summary>参考思路</summary>

开着 profiler 时，每个算子都有额外的记录开销，CPU 侧变慢了 3 倍多，GPU 总是比 CPU 先跑完，所以同步时几乎不用等。line_profiler 只分析了两个函数，对推理部分没有影响，CPU 和 GPU 的速度关系更接近真实情况。不过 line_profiler 本身也有开销，最准确的是用 nsys（开销约百分之几）看 `cudaStreamSynchronize` 的实际耗时（第 04 章）。
</details>

## 练习 4（动手）：读 memory snapshot

在 https://pytorch.org/memory_viz 中打开 `outputs/mem_snapshot_realloc_True.pickle`：
1. 在 "Active Memory Timeline" 中找到周期性出现的大块，点击查看调用栈，确认来自 `pipeline.py` 的哪一行；
2. 切换到 "Allocator State History"，观察 segment 的数量是怎么增长的。

<details><summary>参考思路</summary>

大块来自 `infer()` 中的 `self._scratch = torch.empty(n, ...)`，大小以 5 步为周期在 32/64/96/128/160 MB 之间变化（`(1 + i%5) * 8M` 个 float32）。由于 160 MB 的块不能被拆开给其他用途长期使用，而较小的请求又会优先复用已有的段，segment 的数量会在开头几步快速增长，然后趋于稳定。
</details>

## 练习 5（思考）：怎么在独显平台上做得更多

如果把本章的 demo 搬到 x86 + RTX 显卡上，有哪些分析可以做，而在 Thor 上做不了？

<details><summary>参考思路</summary>

- 算子表中的 `Self CUDA` / `CUDA total` 列：直接看出每个算子的 GPU 耗时，判断是 CPU 侧慢还是 GPU 侧慢；
- trace 中 CPU 算子到 GPU kernel 的关联箭头：看出 launch 延迟和 GPU 空闲时间；
- HTA：`get_idle_time_breakdown`、`get_cuda_kernel_launch_stats`，量化 GPU 空闲的原因；
- TensorBoard 插件的 Overview：GPU 利用率、Tensor Core 使用比例等。
</details>
