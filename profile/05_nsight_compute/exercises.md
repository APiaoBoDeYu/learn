# 第 05 章 练习

## 练习 1（动手）：block size 会改变 Register-bound kernel 的 Occupancy 吗？

在 [03_register_bound.py](03_register_bound.py) 中把 `A_per_thread` 的 `block` 依次改成 32、64、128、256、512，用 ncu 采集 `launch__occupancy_limit_registers`、`sm__maximum_warps_per_active_cycle_pct` 和耗时。

<details><summary>参考思路</summary>

寄存器按 warp 分配：128 regs × 32 线程 = 4096 个/warp，每个 SM 有 65536 个寄存器 → 最多 16 个 warp，所以理论 Occupancy 上限是 16/48 = 33%，与 block size 基本无关。但 block size 会通过"整块分配"产生取整损失：例如 block=512（16 个 warp）时每个 SM 正好放 1 个 block；block=384（12 个 warp）时只能放 1 个，Occupancy 降到 25%。（预期结果，未逐一实测。）用 ncu GUI 的 Occupancy 页面中的 "Impact of Varying Block Size" 图可以直接看到这条曲线。
</details>

## 练习 2（动手）：fp16 的 elementwise 能快多少？

仿照 `scale_vec4` 写一个 `half2` 版本的 `y = a*x`（数据量减半），用 [02_memory_bound.py](02_memory_bound.py) 的方法比较耗时、DRAM 流量和 DRAM 利用率。

<details><summary>参考思路</summary>

Memory-bound kernel 的耗时 ≈ 字节数 / 带宽，所以 fp16 预期接近 2 倍加速，并且 DRAM 利用率不变（仍然接近 100%）。这正是第 07 章"低精度"手段对 Memory-bound 算子生效的原因；对 Compute-bound 的 GEMM，fp16 的收益来自 Tensor Core（06 中实测约 21 倍），机制完全不同。
</details>

## 练习 3（思考）：为什么 eager_3ops 的每个 kernel 在 ncu 里都"很好"？

02 中 eager 的 3 个 kernel 都达到了 DRAM 峰值的 96~99%，ncu 也没有给出重要建议，但总耗时是融合版的约 2.9 倍。请说明：这种问题应该在漏斗的哪一层发现？用什么工具？

<details><summary>参考思路</summary>

ncu 一次只看一个 kernel，判断的是"这个 kernel 是否接近它自己的上限"；它无法知道这三个 kernel 的中间结果本来可以不写回内存。这种"多余的 kernel / 多余的流量"要在 L2/L3 层发现：torch.profiler 的算子表（同一段代码有多个逐元素算子）、nsys 的 kernel 列表（多个 `vectorized_elementwise_kernel` 连续出现），或者用第 00 章的方法手算"必需流量"再与实测对比。修复方法：torch.compile（第 03/07 章）、手写融合 kernel、TensorRT 的层融合（第 06 章）。
</details>

## 练习 4（动手）：用 ncu 分析第 04 章的"拷贝与计算不重叠"

第 04 章 4.5 节发现 Thor 上 H2D 拷贝和 kernel 严格交替。能否用 ncu 解释？

<details><summary>参考思路</summary>

不能。ncu 默认串行化并逐个重放 kernel（`--replay-mode kernel`），拷贝也不在它的分析对象内，所以它**看不到并发**。并发问题应该用 nsys（加上 `--gpu-metrics-devices` 看拷贝期间的 SM 活动和内存带宽）分析。这道题的目的是明确 ncu 的边界：它回答"单个 kernel 为什么慢"，不回答"kernel 之间为什么没有重叠"。
</details>

## 练习 5（动手）：shared memory 的 bank 宽度

把 [cuda/bottlenecks.cu](cuda/bottlenecks.cu) 中 `transpose_smem` 的 tile 类型改成 `double`（并相应修改 load/store），预测 bank conflict 的路数，再用 ncu 验证（`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` 和 wavefronts/指令）。padding 应该加几个元素？

<details><summary>参考思路</summary>

每个 bank 宽 4 字节，一个 double 占 2 个 bank。tile[32][32] 的 double 每行 256 字节 = 64 个 bank 宽度，按列读时同一 warp 的 32 个线程地址相差 256 字节，仍然全部落在同一组 bank 上；一个 warp 请求 32×8 = 256 字节，最少需要 2 个 wavefront，冲突时要 32 个。padding 加 1 个 double（8 字节 = 2 个 bank）即可把相邻行错开。（预期结果，未实测。）
</details>

## 练习 6（动手）：ncu 规则引擎的建议可信吗？

在 03 的报告中，ncu 对 `B_per_thread` 给出了 "TheoreticalOccupancy：预估加速 63%"。对比 `B_split4` 的实测结果，评价这条建议；再看 `B_prefetch` 的报告，ncu 还会给出哪些建议？

<details><summary>参考思路</summary>

`B_split4` 把 Occupancy 从 25% 提高到 82%，实测 0.99 倍，所以这条建议的"预估加速"是在"其他条件不变"的假设下估算的上限，而改变 Occupancy 的手段几乎总会改变其他条件（这里是每个 warp 在两次 load 之间的计算量）。规则引擎的作用是指出值得检查的方向，不能把它的百分比当作预期收益。判断的依据始终是：改动之后**实测**的耗时。
</details>
