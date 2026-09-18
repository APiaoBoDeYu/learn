"""Chapter 3, Example 3: profiling Triton kernels with triton.testing.do_bench,
and a peek at the compiled PTX. Run: python3 03_profiling.py
"""
from common import device_info

import torch
import triton
import triton.testing

from importlib import import_module

# 这两个模块的文件名是 "02_vector_add.py" / "03_matmul_autotune.py"，以数字开头，
# 不是合法的 Python 标识符，没法写成 `import 02_vector_add`（语法错误）。
# 用 importlib.import_module 以字符串形式按模块名导入，就能绕开这个限制，
# 拿到模块对象后照常通过属性访问里面定义的函数/kernel（比如下面的
# vector_add.triton_add、vector_add.add_kernel）。
# 之所以能找到这两个文件，是因为当前工作目录（或脚本所在目录）在 sys.path 里，
# 而这两个文件恰好和 03_profiling.py 放在同一个 code/ 目录下。
vector_add = import_module("02_vector_add")
matmul = import_module("03_matmul_autotune")


if __name__ == "__main__":
    device_info()

    # --- triton.testing.do_bench: the library's own benchmarking helper ---
    # It does warmup + repeated timing with cuda events, and can report
    # median/min/max or a custom quantile — similar spirit to our bench_ms
    # in common.py but built into Triton for convenience.
    n = 2**20
    x = torch.rand(n, device="cuda")
    y = torch.rand(n, device="cuda")

    ms = triton.testing.do_bench(lambda: vector_add.triton_add(x, y))  # 内部会自动 warmup + 用 cuda event 重复计时，返回中位数耗时（ms）
    print(f"do_bench vector_add: {ms:.4f} ms")

    ms_torch = triton.testing.do_bench(lambda: x + y)  # 同样的方式给 torch 原生加法计时，方便对比
    print(f"do_bench torch add : {ms_torch:.4f} ms")

    # --- Inspecting generated PTX for a compiled kernel ------------------
    # Every JITFunction caches compiled kernels keyed by input signature, in
    # `.device_caches` (triton 3.6 API). After at least one call, this holds
    # CompiledKernel objects that expose the generated PTX/SASS-adjacent
    # assembly for inspection — the Triton equivalent of looking at
    # nvcc-generated PTX for CUDA C++.
    out = torch.empty_like(x)
    grid = (triton.cdiv(n, 1024),)
    vector_add.add_kernel[grid](x, y, out, n, BLOCK_SIZE=1024)  # 先手动调用一次 kernel，确保它被编译过、缓存里有内容可看

    device_id = next(iter(vector_add.add_kernel.device_caches))  # 取第一个（当前用到的）GPU 设备 id
    kernels_by_signature = vector_add.add_kernel.device_caches[device_id][0]  # 同一个 kernel 针对不同参数签名（dtype/BLOCK_SIZE等）会编译出多个版本，这里按签名缓存
    compiled_kernel = next(iter(kernels_by_signature.values()))  # 取其中一个已编译的版本（这里只调用过一次，所以只有一个）
    ptx = compiled_kernel.asm["ptx"]  # 取出编译产物里的 PTX 汇编文本（NVIDIA GPU 的中间汇编语言，类似 nvcc 编译 CUDA C++ 生成的 PTX）
    print(f"\nGenerated PTX for add_kernel (first 15 lines of {len(ptx.splitlines())}):")
    for line in ptx.splitlines()[:15]:
        print(" ", line)

    print("\nAvailable compiled artifacts:", list(compiled_kernel.asm.keys()))  # 打印还能看到哪些其他编译产物（比如 ttir/ttgir/cubin 等中间表示）
