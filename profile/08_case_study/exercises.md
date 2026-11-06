# 第 08 章 练习

## 练习 1（动手）：换一种修复顺序

把 [case.py](case.py) 中 `STAGES` 的顺序改成"先 fp16、再 TensorRT、最后处理 CPU 侧"（先优化推理），重新运行 [02_optimize.py](02_optimize.py)。最终结果会一样吗？每一步的"本步加速"呢？哪种顺序更能体现诊断的价值？

<details><summary>参考思路</summary>

最终配置相同，最终延迟应该接近（约 0.75 ms）。但每一步的加速比会完全不同：先换 TensorRT 时，推理从约 4.2 ms 降到约 0.4 ms，但端到端只从 10.3 降到约 6.5 ms，因为后处理和预处理还有约 6.7 ms（Amdahl 定律）。按诊断顺序做，每一步都在处理"当前最大的一块"，单步收益更大，也更容易判断每一步是否有效。（预期结果，未实测。）
</details>

## 练习 2（动手）：把整个 step 捕获成一个 CUDA Graph

S9 中预处理（H2D + 插值 + 归一化）、推理（TensorRT）、后处理（向量化）都在 GPU 上，只有最后一次 D2H 和 `.cpu().numpy()` 需要同步。尝试把 H2D 之后到 D2H 之前的部分用 `torch.cuda.graph` 捕获成一个图，测量端到端延迟和 nsys 中的 GPU busy。

<details><summary>参考思路</summary>

需要：所有中间张量预分配、没有 `.item()`、`prev_action` 用原地更新（`copy_`）而不是重新赋值、TensorRT 的 execute 也在同一个捕获里（第 06 章 05_compare.py 的做法）。剩下的 CPU 开销只有一次 H2D、一次 graph replay、一次 D2H，预期能再省掉相当一部分 CPU 调度时间。（预期结果，未实测。）
</details>

## 练习 3（动手）：两帧流水

让第 i+1 帧的预处理（CPU 上的 `capture` + H2D）与第 i 帧的推理重叠：用两个 CUDA stream 和双缓冲。测量吞吐（帧/秒）和单帧延迟的变化。结合第 04 章 4.8 的结论，H2D 拷贝的大小应该怎么选？

<details><summary>参考思路</summary>

吞吐有可能提高（CPU 侧的工作和 GPU 工作重叠了），但单帧延迟不会降低。Thor 上 ≥256 KB 的拷贝由 SM 执行，会和推理抢 SM；<256 KB 的拷贝走 Copy Engine，能并发，但带宽低、尾延迟大（REPORT.md 第 7 节）。本例原始帧是 900 KB，由 SM 执行，拷贝本身只要约 40 μs，可以接受它和推理交替执行，重点是让 CPU 侧的工作重叠。（预期结果，未实测。）
</details>

## 练习 4（写报告）：用模板分析你自己的模型

选一个自己的模型（例如 `../vla` 中的真实 VLA），按 [report_template.md](report_template.md) 完成一份报告：至少包含 L0~L3 的诊断、3 个以上的优化步骤（含一个无效的尝试）、每一步的 p50/p99、能耗和精度。

<details><summary>参考思路</summary>

真实模型比 ToyVLA 大得多，ncu 一定要用过滤（第 05 章 5.2），nsys 只采稳态（第 04 章 4.1）。prefill 和 decode 要分开度量（第 07 章 7.5）。分享 nsys 报告之前，检查其中的环境变量（第 04 章常见坑 11）。
</details>
