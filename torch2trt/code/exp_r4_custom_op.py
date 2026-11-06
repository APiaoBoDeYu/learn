"""R4（torch.export + Torch-TensorRT）遇到 torch.library 自定义算子时的三种处理方式。

把 erfinv 包装成自定义算子 `mylib::my_erfinv`，分别用三种方式转换：

  fallback   只注册算子。没有 converter，该算子留在 PyTorch 执行（部分回退）。
  cpp        加载 C++ 插件 .so + 注册 converter，算子进入 TRT 引擎；
             整图都在 TRT 中，因此还能导出独立的 .plan 引擎文件。
  auto_qdp   算子用 Triton 实现，一行 plugins.custom_op 自动生成插件 + converter。

“模型 -> TRT 引擎”的转换发生在两处：
  1. routes.build_dynamo_trt：export + tt.dynamo.compile，得到内嵌 TRT 引擎的
     GraphModule（引擎只在内存中）。三个变体都会走。
  2. export_plan：convert_exported_program_to_serialized_trt_engine，得到引擎字节流
     并写成 .plan 文件。只有 cpp 变体走（要求整图无回退）。

每个变体单独开子进程：算子 / converter / 插件注册表都是进程级全局状态，不能重复注册。

尝试过的方案（Thor, TRT 10.14, Torch-TensorRT 2.9, Triton 3.5, b=32, fp16）：
  方案                           导出 .plan  新进程纯 TRT 运行  纯 TRT 执行耗时
  fallback                       失败        —                  —
  cpp（C++ IPluginV3 .so）       成功        成功（先加载 .so） 0.029 ms
  auto_qdp（QDP JIT）            成功        失败：插件只在 Python 进程中注册，
                                                   运行时回调 Python -> Triton
  手写 QDP AOT（trtp.aot_impl    成功        成功，无需 Python   0.214 ms
    返回 Triton 编译的 PTX）                 / Triton / .so
  手写 QDP AOT，返回 cubin       成功        失败：onShapeChange status -1
    （Triton cubin 或 ptxas 编译）            （TRT 10.14 只支持 PTX）
  AOT 的坑：Triton 3.5 会给 PTX 追加 2 个 scratch 指针参数，TRT 只传 x, n, y，
  会导致段错误，需要从 PTX 中删掉这两个参数；kernel 内不能用 Python 列表循环。
  AOT 慢的原因：nsys 显示每次推理都调用一次 cuModuleLoadData 重新加载 PTX 模块
  （cpp 在整个运行期间只调用 3 次）。实验代码在 /tmp/aot/，没有并入本脚本。

结论：R4（torch.export -> Torch-TensorRT dynamo_trt）遇到不支持的算子时，
最优做法是写 C++ plugin + converter：最快、整图进 engine、.plan 可以脱离 Python 部署。

用法：
  bash plugins/build.sh                     # 编译 artifacts/liberfinv_plugin.so
  python3 exp_r4_custom_op.py               # 依次跑三个变体
  python3 exp_r4_custom_op.py --variant cpp
"""
import argparse
import subprocess
import sys

import torch

import common
import models
import routes

PLUGIN_SO = common.ARTIFACTS / "liberfinv_plugin.so"
VARIANTS = ["fallback", "cpp", "auto_qdp"]


# ---- 第 1 步：注册自定义算子（fallback 变体只需要这一步） ----
def define_op(impl):
    @torch.library.custom_op("mylib::my_erfinv", mutates_args=())
    def my_erfinv(x: torch.Tensor) -> torch.Tensor:
        return impl(x)

    # 必须提供：torch.export 用 FakeTensor 追踪，需要知道输出的 shape/dtype
    @my_erfinv.register_fake
    def _(x):
        return torch.empty_like(x)


class CustomOpModel(models.NoisyActivation):
    """与模型库的 `customop` 结构/权重相同，只是 erfinv 换成了自定义算子。"""

    def forward(self, x):
        x = self.fc1(x)
        x = torch.ops.mylib.my_erfinv(x.clamp(-0.99, 0.99))
        return self.fc2(x)


# ---- 第 2a 步：为算子注册 converter，映射到 C++ 插件 ----
def register_cpp_converter():
    import ctypes

    import tensorrt as trt
    from torch_tensorrt.dynamo.conversion import dynamo_tensorrt_converter
    from torch_tensorrt.dynamo.conversion.converter_utils import get_trt_tensor

    ctypes.CDLL(str(PLUGIN_SO))   # 加载 .so 时会把 "ErfInv" 插件注册进 TRT

    @dynamo_tensorrt_converter(torch.ops.mylib.my_erfinv.default,
                               supports_dynamic_shapes=True)
    def my_erfinv_converter(ctx, target, args, kwargs, name):
        creator = trt.get_plugin_registry().get_creator("ErfInv", "1", "")
        plugin = creator.create_plugin("ErfInv", trt.PluginFieldCollection([]),
                                       trt.TensorRTPhase.BUILD)
        x = get_trt_tensor(ctx, args[0], f"{name}_x")
        layer = ctx.net.add_plugin_v3([x], [], plugin)
        layer.name = name
        return layer.get_output(0)


# ---- 第 2b 步：Triton 实现（auto_qdp 靠追踪这个 kernel 生成插件） ----
def triton_erfinv():
    import triton
    import triton.language as tl

    @triton.jit
    def erfinv_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
        # Giles 单精度 erfinv 近似：按 w 分两段多项式
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        x = tl.load(x_ptr + offs, mask=m).to(tl.float32)
        w = -tl.log((1.0 - x) * (1.0 + x))
        t1 = w - 2.5
        p1 = 2.81022636e-08
        p1 = 3.43273939e-07 + p1 * t1
        p1 = -3.5233877e-06 + p1 * t1
        p1 = -4.39150654e-06 + p1 * t1
        p1 = 0.00021858087 + p1 * t1
        p1 = -0.00125372503 + p1 * t1
        p1 = -0.00417768164 + p1 * t1
        p1 = 0.246640727 + p1 * t1
        p1 = 1.50140941 + p1 * t1
        t2 = tl.sqrt(w) - 3.0
        p2 = -0.000200214257
        p2 = 0.000100950558 + p2 * t2
        p2 = 0.00134934322 + p2 * t2
        p2 = -0.00367342844 + p2 * t2
        p2 = 0.00573950773 + p2 * t2
        p2 = -0.0076224613 + p2 * t2
        p2 = 0.00943887047 + p2 * t2
        p2 = 1.00167406 + p2 * t2
        p2 = 2.83297682 + p2 * t2
        p = tl.where(w < 5.0, p1, p2)
        tl.store(y_ptr + offs, (p * x).to(y_ptr.dtype.element_ty), mask=m)

    def impl(x):
        y = torch.empty_like(x)
        erfinv_kernel[(triton.cdiv(x.numel(), 256),)](x, y, x.numel(), BLOCK=256)
        return y
    return impl


# ---- 运行一个变体 ----
def run_variant(variant, batch):
    if variant == "auto_qdp":
        import torch_tensorrt as tt
        define_op(triton_erfinv())
        # 一行自动生成 QDP 插件 + converter
        tt.dynamo.conversion.plugins.custom_op("mylib::my_erfinv",
                                               supports_dynamic_shapes=True)
    else:
        define_op(torch.erfinv)
        if variant == "cpp":
            register_cpp_converter()

    # 被转换的 fp16 模型，以及作为精度基准的 fp32 原始 customop 模型
    base, inputs = models.get("customop", batch=batch)
    model = CustomOpModel().eval().to("cuda", torch.half)
    model.load_state_dict(base.state_dict())
    m32, in32 = models.get("customop", batch=batch, dtype=torch.float32)
    with torch.no_grad():
        ref = m32(*in32).float().cpu()

    # ★ 主转换：export + tt.dynamo.compile -> 内嵌 TRT 引擎的 GraphModule
    runnable, meta = routes.build_dynamo_trt(model, inputs)
    print(f"  TRT 子图={meta['trt_subgraphs']}  回退 PyTorch 子图={meta['fallback_subgraphs']}")

    with torch.no_grad():
        common.compare(runnable(*inputs).float().cpu(), ref, f"r4_{variant}")
    print(f"  median={common.bench_ms(lambda: runnable(*inputs))['median_ms']:.3f}ms")

    if variant == "cpp":
        export_plan(model, inputs, ref, batch)


def export_plan(model, inputs, ref, batch):
    """★ 导出独立引擎文件：整图无回退时才能成功，之后可用纯 TensorRT 加载。"""
    import tensorrt as trt
    import torch_tensorrt as tt

    ep = torch.export.export(model, inputs)
    blob = tt.dynamo.convert_exported_program_to_serialized_trt_engine(
        ep, inputs=list(inputs),
        enabled_precisions={torch.half, torch.float},
        min_block_size=1, truncate_double=True, offload_module_to_cpu=True)
    model.to("cuda")    # offload_module_to_cpu 会把模型留在 CPU，移回 GPU
    plan_path = common.ARTIFACTS / f"customop_b{batch}_r4_cpp.plan"
    plan_path.write_bytes(blob)
    print(f"  plan: {plan_path}")

    # 用纯 TensorRT 运行 .plan，验证精度
    got = routes.TrtRunner(blob, trt.Logger(trt.Logger.ERROR))(*inputs)
    got = got[0] if isinstance(got, (tuple, list)) else got
    common.compare(got.float().cpu(), ref, "r4_cpp_plan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=VARIANTS)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    if args.variant:
        run_variant(args.variant, args.batch)
        return

    # 每个变体一个子进程，避免全局注册表互相污染
    for v in VARIANTS:
        print(f"\n--- {v} ---", flush=True)
        # 相当于在终端执行了好几次命令
        subprocess.run([sys.executable, __file__, "--variant", v, "--batch", str(args.batch)])


if __name__ == "__main__":
    main()
