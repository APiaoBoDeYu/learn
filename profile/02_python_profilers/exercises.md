# 第 02 章 练习

## 练习 1（动手）：修好预处理

把 `_preprocess_python` 改成 numpy 向量化的版本（不用 Python 循环），用 line_profiler 对比改前改后，并用 `benchmark` 测量整体延迟的变化。

<details><summary>参考思路</summary>

```python
x = (small.astype(np.float32) / 255.0 - MEAN) / STD    # (224,224,3)，自动广播
x = np.ascontiguousarray(x.transpose(2, 0, 1))[None]   # (1,3,224,224)
```
一次处理整张图，numpy 只调用几次，耗时从约 3 ms 降到 0.1~0.2 ms 量级。更进一步是把归一化放到 GPU 上做（`PipelineConfig.fast()` 中的 `_preprocess_gpu`），这样 CPU 只需要拷贝 uint8 数据。
</details>

## 练习 2（思考）：为什么 `.item()` 每次要 0.2 ms？

line_profiler 显示第 150 行每次执行 0.2 ms，第 149 行每次只要 0.03 ms。`.item()` 本身只拷贝 4 个字节，时间花在哪里了？

<details><summary>参考思路</summary>

`.item()` 需要把 GPU 上的值拷到 CPU，必须等它前面所有已提交的 kernel 都执行完（隐式同步）。第 149 行在 GPU 上排了一串小 kernel，CPU 提交得比 GPU 执行得快，`.item()` 就要等这些 kernel 全部跑完。另外，第一个维度的 `.item()` 还要等推理阶段剩下的 GPU 工作。所以 0.2 ms 里绝大部分是等 GPU 的时间。
验证方法：在 `.item()` 之前加一个 `torch.cuda.synchronize()`，并用 line_profiler 分别统计，等待时间会转移到 synchronize 那一行上。
</details>

## 练习 3（动手）：py-spy 抓"卡住"的进程

写一个脚本，在第 5 秒时故意死锁（例如 `threading.Lock()` 获取两次），用 `workload.py` 里的 `allow_ptrace()` 允许追踪，然后用 `py-spy dump --pid` 找到卡住的位置。

<details><summary>参考思路</summary>

dump 的输出会显示主线程停在 `lock.acquire` 那一行，线程状态是 idle（没有持有 GIL）。加上 `--locals` 还能看到局部变量。这是排查"程序没反应"最快的办法，不需要重启进程。
</details>

## 练习 4（动手）：VizTracer 的多进程

DataLoader 设置 `num_workers=4` 时，预处理在子进程中执行。用命令行 `viztracer -o mp.json your_script.py` 追踪一个带 DataLoader 的小脚本（VizTracer 1.1 用命令行启动时默认会追踪 multiprocessing 创建的子进程，不需要时可以加 `--ignore_multiprocess` 关闭），然后在 Perfetto 中查看各个 worker 的时间线。

<details><summary>参考思路</summary>

Perfetto 中每个进程是一组单独的轨道，可以看到 worker 是否并行工作、主进程是否在等数据（主进程的 `__next__` 很长，说明数据加载是瓶颈）。如果 worker 的时间线中间有大段空白，说明 `prefetch_factor` 或 `num_workers` 不够。
</details>

## 练习 5（思考）：选工具

下面的场景分别用哪个工具？
1. 线上推理服务偶尔卡住 2 秒，不能重启；
2. 想知道预处理函数中哪一行最慢；
3. 想看 ROS 节点里三个回调线程的执行顺序和重叠情况；
4. 想知道模型推理中 GPU 的时间分布。

<details><summary>参考思路</summary>

1. py-spy dump/top（无侵入，可以 attach），或者 `py-spy record --pid` 采一段时间；
2. line_profiler；
3. VizTracer（多线程时间线）；
4. Python profiler 都不行，要用 nsys（第 04 章）。在独显平台上也可以用 torch.profiler。
</details>
