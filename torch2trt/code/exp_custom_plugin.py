"""Experiment: the two escape hatches chapter 05 skipped -- an ONNX custom op on
the export side, a TensorRT plugin on the build side.

Chapter 05 recorded `customop / onnx_trt` as "build failed" and moved on. That
is what happens if you stop at the first error, but it is not the end of the
ONNX route: the standard production answer is

  1. eager -> ONNX : teach the exporter to emit a node for the op it does not
                     know (a custom-domain node, or a decomposition into
                     standard ONNX ops), and
  2. ONNX -> engine: give the TensorRT parser an implementation for that node
                     (a plugin registered under the same name).

This script measures every practical combination of those two steps on the
same `customop` model (fc1 -> clamp -> erfinv -> fc2), so the comparison with
dynamo_trt's partial fallback and AOTInductor is apples to apples.

Variants (each runs in its own subprocess so one variant's plugin registration
cannot leak into another -- the TRT plugin registry is process-global):

  onnx_no_plugin       custom::ErfInv node, nothing registered. The control:
                       shows that a custom ONNX op alone is not enough.
  onnx_cpp_plugin      custom node + C++ IPluginV3 (.so), Python builder.
  trtexec_cpp_plugin   same ONNX + .so, built by `trtexec --staticPlugins`.
  onnx_dynamo_cpp      custom node emitted by the dynamo=True exporter
                       (custom_translation_table) + the same .so.
  onnx_qdp_jit         custom node + Python plugin (tensorrt.plugin, Triton
                       kernel in @trtp.impl). Fastest to write; engine needs
                       Python at runtime.
  onnx_qdp_aot         custom node + Python plugin with @trtp.aot_impl (PTX
                       embedded in the engine). Python only at build time.
  onnx_decomposed      no custom node: the export-side symbolic rewrites
                       erfinv into ~70 standard ONNX ops. No plugin at all.
  eager / aoti / dynamo_trt_fallback
                       baselines, re-measured in the same session.
  dynamo_trt_cpp       the same C++ plugin, but wired into Torch-TensorRT as a
                       converter, so erfinv no longer falls back to PyTorch.

Usage:
  bash plugins/build.sh                 # builds artifacts/liberfinv_plugin.so
  python3 exp_custom_plugin.py          # all variants
  python3 exp_custom_plugin.py --batch 8192      # compute-bound regime
  python3 exp_custom_plugin.py --variant onnx_cpp_plugin   # one, in-process
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

import common
import models

ART = common.ARTIFACTS
PLUGIN_SO = ART / "liberfinv_plugin.so"
CODE_DIR = Path(__file__).resolve().parent

# Baselines re-measured in the same session so the comparison does not lean on
# chapter 06's numbers from a different run.
BASELINES = ["eager", "aoti", "dynamo_trt_fallback"]
VARIANTS = BASELINES + ["onnx_no_plugin", "onnx_cpp_plugin", "trtexec_cpp_plugin",
            "onnx_dynamo_cpp", "onnx_qdp_jit", "onnx_qdp_aot",
            "onnx_decomposed", "dynamo_trt_cpp"]


class Skip(Exception):
    """A variant that ran but legitimately produced no runnable (the control)."""


# --------------------------------------------------------------------------
# Step 1: eager -> ONNX
# --------------------------------------------------------------------------
def export_custom_node_legacy(model, inputs, path):
    """TorchScript-based exporter: register a symbolic for aten::erfinv.

    The symbolic receives the ONNX graph builder `g` and returns whatever nodes
    should stand in for the op. Here: one node in a private domain. The domain
    name is arbitrary, but it must be declared in `custom_opsets` or onnx.checker
    rejects the file.
    """
    def erfinv_symbolic(g, x):
        return g.op("custom::ErfInv", x).setType(x.type())

    torch.onnx.register_custom_op_symbolic("aten::erfinv", erfinv_symbolic, 17)
    torch.onnx.export(model, inputs, str(path), opset_version=17,
                      input_names=["in0"], output_names=["out"],
                      custom_opsets={"custom": 1}, dynamo=False)


def export_custom_node_dynamo(model, inputs, path):
    """torch.export-based exporter: a custom_translation_table entry.

    Two things the docs undersell:
      - onnxscript resolves `custom.ErfInv` by looking up an ONNX OpSchema, so a
        schema has to be registered first, or translation fails with
        "No schema registered for 'ErfInv' version '1' and domain 'custom'".
      - Wrapping the function in @onnxscript.script fails differently (the
        traced output loses its shape and the *next* node, fc2, errors out).
        A plain Python function that calls the op works.
    """
    import onnx.defs as D
    import onnxscript

    if not D.has("ErfInv", "custom"):
        D.register_schema(D.OpSchema(
            "ErfInv", "custom", 1,
            inputs=[D.OpSchema.FormalParameter("X", "T")],
            outputs=[D.OpSchema.FormalParameter("Y", "T")],
            type_constraints=[("T", ["tensor(float)", "tensor(float16)"], "")]))
    custom = onnxscript.values.Opset("custom", 1)

    def erfinv_onnx(x):
        return custom.ErfInv(x)

    torch.onnx.export(model, inputs, str(path),
                      input_names=["in0"], output_names=["out"], dynamo=True,
                      custom_translation_table={torch.ops.aten.erfinv.default: erfinv_onnx})


# Giles (2010) single-precision erfinv: two polynomial branches. Only standard
# ONNX ops are needed, which is the whole point of this variant.
_GILES_A = [2.81022636e-08, 3.43273939e-07, -3.5233877e-06, -4.39150654e-06,
            0.00021858087, -0.00125372503, -0.00417768164, 0.246640727, 1.50140941]
_GILES_B = [-0.000200214257, 0.000100950558, 0.00134934322, -0.00367342844,
            0.00573950773, -0.0076224613, 0.00943887047, 1.00167406, 2.83297682]


def export_decomposed(model, inputs, path):
    """Same symbolic hook, but emit standard ops instead of a custom node.

    This is "rewrite the op" done at the export boundary instead of in the
    model source: the nn.Module is untouched, and the ONNX file needs nothing
    beyond opset 17. Arithmetic runs in fp32 (erfinv near +-1 is too steep for
    fp16), with casts at both ends.
    """
    def c(g, v):
        return g.op("Constant", value_t=torch.tensor(v, dtype=torch.float32))

    def horner(g, coeffs, t):
        p = c(g, coeffs[0])
        for k in coeffs[1:]:
            p = g.op("Add", c(g, k), g.op("Mul", p, t))
        return p

    def erfinv_symbolic(g, x):
        xf = g.op("Cast", x, to_i=1)                      # -> float32
        one = c(g, 1.0)
        w = g.op("Neg", g.op("Log", g.op("Mul", g.op("Sub", one, xf),
                                         g.op("Add", one, xf))))
        p1 = horner(g, _GILES_A, g.op("Sub", w, c(g, 2.5)))
        p2 = horner(g, _GILES_B, g.op("Sub", g.op("Sqrt", w), c(g, 3.0)))
        p = g.op("Where", g.op("Less", w, c(g, 5.0)), p1, p2)
        return g.op("Cast", g.op("Mul", p, xf), to_i=10)  # -> float16

    torch.onnx.register_custom_op_symbolic("aten::erfinv", erfinv_symbolic, 17)
    torch.onnx.export(model, inputs, str(path), opset_version=17,
                      input_names=["in0"], output_names=["out"], dynamo=False)


# --------------------------------------------------------------------------
# Step 2: plugins
# --------------------------------------------------------------------------
def load_cpp_plugin():
    """Loading the .so is the whole registration: REGISTER_TENSORRT_PLUGIN runs
    a static constructor that adds the creator to the global registry."""
    import ctypes
    if not PLUGIN_SO.exists():
        raise RuntimeError(f"{PLUGIN_SO} missing -- run `bash plugins/build.sh` first")
    ctypes.CDLL(str(PLUGIN_SO))


def register_qdp_jit():
    """Python plugin, JIT flavour: TensorRT calls back into this Python function
    on every enqueue. A Triton kernel does the work."""
    from typing import Tuple

    import tensorrt.plugin as trtp
    import triton
    import triton.language as tl

    @triton.jit
    def erfinv_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
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

    @trtp.register("custom::ErfInv")
    def erfinv_desc(x: trtp.TensorDesc) -> Tuple[trtp.TensorDesc]:
        return x.like()

    @trtp.impl("custom::ErfInv")
    def erfinv_impl(x: trtp.Tensor, outputs: Tuple[trtp.Tensor], stream: int) -> None:
        xt = torch.as_tensor(x, device="cuda")
        yt = torch.as_tensor(outputs[0], device="cuda")
        n = xt.numel()
        with torch.cuda.stream(torch.cuda.ExternalStream(stream)):
            erfinv_kernel[(triton.cdiv(n, 1024),)](xt, yt, n, BLOCK=1024)


_AOT_CU = r'''
#include <cuda_fp16.h>
extern "C" __global__ void erfinv_f16(const __half* x, int n, __half* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(erfinvf(__half2float(x[i])));
}
extern "C" __global__ void erfinv_f32(const float* x, int n, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = erfinvf(x[i]);
}
'''


def register_qdp_aot():
    """Python plugin, AOT flavour: Python only describes the kernel and its
    launch geometry at build time; the PTX is embedded in the engine.

    Why the kernel is nvcc-compiled CUDA C, not Triton as in NVIDIA's example:
    Triton 3.5 appends two hidden scratch-pointer parameters to every kernel
    (visible in its PTX `.entry`), and TensorRT 10.14's AOT launcher does not
    pass them -- the engine builds fine and segfaults on the first enqueue.
    Plain `extern "C"` CUDA has exactly the signature you write.
    """
    import tempfile
    from typing import Tuple, Union

    import tensorrt as trt
    import tensorrt.plugin as trtp

    d = Path(tempfile.mkdtemp())
    (d / "k.cu").write_text(_AOT_CU)
    subprocess.check_call(["nvcc", "-ptx", "-arch=sm_110a", str(d / "k.cu"),
                           "-o", str(d / "k.ptx")])
    ptx = (d / "k.ptx").read_text()

    @trtp.register("custom::ErfInv")
    def erfinv_desc(x: trtp.TensorDesc) -> Tuple[trtp.TensorDesc]:
        return x.like()

    @trtp.aot_impl("custom::ErfInv")
    def erfinv_aot(x: trtp.TensorDesc, outputs: Tuple[trtp.TensorDesc], tactic: int
                   ) -> Tuple[Union[str, bytes], Union[str, bytes],
                              trtp.KernelLaunchParams, trtp.SymExprs]:
        name = "erfinv_f16" if x.dtype == trt.float16 else "erfinv_f32"
        n = x.shape_expr.numel()
        lp = trtp.KernelLaunchParams()
        lp.grid_x = trtp.cdiv(n, 256)
        lp.block_x = 256
        extra = trtp.SymIntExprs(1)       # scalar args go between inputs and outputs
        extra[0] = trtp.SymInt32(n)
        return name, ptx, lp, extra


# --------------------------------------------------------------------------
# Build helpers
# --------------------------------------------------------------------------
def build_engine_py(onnx_path, plan_path):
    """Same builder code as routes.build_onnx_trt, plus the layer-type census
    that proves the plugin (not some fallback) is what ended up in the network."""
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)
    # parse_from_file, not parse(bytes): the dynamo=True exporter writes weights
    # to a sidecar `<name>.onnx.data` by default (external_data=True), and
    # parse(bytes) has no directory to resolve it against -- it fails with
    # "Failed to import initializer", which looks like a plugin problem but isn't.
    if not parser.parse_from_file(str(onnx_path)):
        errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise Skip("onnx parse failed: " + "; ".join(errs))
    plugin_layers = [network.get_layer(i).name for i in range(network.num_layers)
                     if "PLUGIN" in str(network.get_layer(i).type)]
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("builder returned no engine")
    Path(plan_path).write_bytes(plan)
    return bytes(plan), {"plugin_layers": plugin_layers,
                         "network_layers": network.num_layers}


def trtexec_gpu_ms(plan_path, static_plugin=False):
    """Load the engine in trtexec: a C++ process with no Python.

    Two answers in one call: whether the engine is deployable without Python
    at all (rc), and its pure GPU compute time with no TrtRunner overhead.
    """
    cmd = ["trtexec", f"--loadEngine={plan_path}"]
    if static_plugin:
        cmd.append(f"--staticPlugins={PLUGIN_SO}")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    ok = proc.returncode == 0 and "PASSED" in proc.stdout
    gpu_ms = None
    for line in proc.stdout.splitlines():
        if "GPU Compute Time:" in line and "median =" in line:
            gpu_ms = float(line.split("median =")[1].split("ms")[0])
    err = None
    if not ok:
        err = next((l for l in (proc.stdout + proc.stderr).splitlines()
                    if "Error" in l or "rror[" in l), proc.stderr[-200:])
    return {"python_free_load": ok, "trtexec_gpu_ms": gpu_ms,
            "trtexec_error": err[:300] if err else None}


# --------------------------------------------------------------------------
# Variants
# --------------------------------------------------------------------------
def run_variant(variant, batch=32):
    """Build + check one variant in the current process. Returns a record."""
    import routes
    import tensorrt as trt

    model, inputs = models.get("customop", batch=batch)
    m32, in32 = models.get("customop", batch=batch, dtype=torch.float32)
    with torch.no_grad():
        ref = m32(*in32).float().cpu()

    onnx_path = ART / f"customop_b{batch}_{variant}.onnx"
    plan_path = ART / f"customop_b{batch}_{variant}.plan"
    rec = {"variant": variant, "batch": batch}
    t0 = time.perf_counter()

    if variant in BASELINES:
        route = "dynamo_trt" if variant == "dynamo_trt_fallback" else variant
        runnable, meta = routes.ROUTES[route](model, inputs, name=f"customop_b{batch}")
        rec.update({k: v for k, v in meta.items() if isinstance(v, (int, float))})
        rec["python_free_load"] = variant == "aoti"
    elif variant == "dynamo_trt_cpp":
        load_cpp_plugin()
        from torch_tensorrt.dynamo.conversion import dynamo_tensorrt_converter
        from torch_tensorrt.dynamo.conversion.converter_utils import get_trt_tensor

        # A converter is Torch-TensorRT's equivalent of the ONNX parser's
        # plugin lookup: "when you see aten.erfinv, add this layer".
        @dynamo_tensorrt_converter(torch.ops.aten.erfinv.default,
                                   supports_dynamic_shapes=True)
        def erfinv_converter(ctx, target, args, kwargs, name):
            creator = trt.get_plugin_registry().get_creator("ErfInv", "1", "")
            plugin = creator.create_plugin("ErfInv", trt.PluginFieldCollection([]),
                                           trt.TensorRTPhase.BUILD)
            x = get_trt_tensor(ctx, args[0], f"{name}_x")
            layer = ctx.net.add_plugin_v3([x], [], plugin)
            layer.name = name
            return layer.get_output(0)

        runnable, meta = routes.build_dynamo_trt(model, inputs)
        rec.update(build_s=meta["build_s"], trt_subgraphs=meta["trt_subgraphs"],
                   fallback_subgraphs=meta["fallback_subgraphs"],
                   python_free_load=False)
    else:
        if variant == "onnx_dynamo_cpp":
            export_custom_node_dynamo(model, inputs, onnx_path)
        elif variant == "onnx_decomposed":
            export_decomposed(model, inputs, onnx_path)
        else:
            export_custom_node_legacy(model, inputs, onnx_path)
        rec["onnx_export_s"] = time.perf_counter() - t0

        import onnx
        g = onnx.load(str(onnx_path)).graph
        rec["onnx_nodes"] = len(g.node)
        rec["onnx_custom_nodes"] = [f"{n.domain}::{n.op_type}" for n in g.node if n.domain]

        if variant in ("onnx_cpp_plugin", "onnx_dynamo_cpp", "trtexec_cpp_plugin"):
            load_cpp_plugin()
        elif variant == "onnx_qdp_jit":
            register_qdp_jit()
        elif variant == "onnx_qdp_aot":
            register_qdp_aot()

        tb = time.perf_counter()
        if variant == "trtexec_cpp_plugin":
            proc = subprocess.run(
                ["trtexec", f"--onnx={onnx_path}", f"--saveEngine={plan_path}",
                 "--fp16", f"--staticPlugins={PLUGIN_SO}"],
                capture_output=True, text=True, timeout=1800)
            if proc.returncode != 0:
                raise RuntimeError(proc.stdout[-400:])
            plan = plan_path.read_bytes()
        else:
            plan, info = build_engine_py(onnx_path, plan_path)
            rec.update(info)
        rec["build_s"] = time.perf_counter() - tb
        rec["engine_mb"] = common.engine_size_mb(plan_path)
        runnable = routes.TrtRunner(plan, trt.Logger(trt.Logger.ERROR))
        rec.update(trtexec_gpu_ms(plan_path, static_plugin="cpp" in variant))

    timing = common.bench_ms(lambda: runnable(*inputs))
    with torch.no_grad():
        got = runnable(*inputs)
    got = (got[0] if isinstance(got, (tuple, list)) else got).float().cpu()
    rec.update(outcome="built", **timing, **common.compare(got, ref, variant))
    print(f"  median={timing['median_ms']:.3f}ms")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=VARIANTS)
    ap.add_argument("--batch", type=int, default=32,
                    help="b=32 is launch-overhead bound; try 8192 for compute-bound")
    ap.add_argument("--json", action="store_true",
                    help="(internal) print the record as the last stdout line")
    args = ap.parse_args()

    if args.variant:
        try:
            rec = run_variant(args.variant, args.batch)
        except Skip as e:
            print(f"  REFUSED: {str(e)[:300]}")
            rec = {"variant": args.variant, "batch": args.batch,
                   "outcome": "build_failed", "error": str(e)[:400]}
        if args.json:
            print("@@RESULT " + json.dumps(rec, default=str))
        return

    common.device_info()
    if not PLUGIN_SO.exists():
        subprocess.check_call(["bash", str(CODE_DIR / "plugins" / "build.sh")])

    rows = []
    for v in VARIANTS:
        print(f"\n--- {v} ---", flush=True)
        proc = subprocess.run([sys.executable, __file__, "--variant", v, "--json",
                               "--batch", str(args.batch)],
                              capture_output=True, text=True, cwd=CODE_DIR)
        lines = proc.stdout.splitlines()
        res = [l for l in lines if l.startswith("@@RESULT ")]
        for l in lines:
            if l.startswith(("[PASS]", "[DIFF]", "  ")):
                print(l)
        if res:
            rec = json.loads(res[-1][len("@@RESULT "):])
        else:
            tail = "\n".join(proc.stderr.strip().splitlines()[-3:])
            print(f"  CRASHED: {tail}")
            rec = {"variant": v, "batch": args.batch, "outcome": "crashed", "error": tail[:400]}
        rows.append(rec)
        common.save_result("plugin_experiment", rec)

    print(f"\n\n### custom-op / plugin verdict (customop, b={args.batch}, fp16)\n")
    print("| variant | outcome | median ms | trtexec GPU ms | python-free | cos | notes |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        ms = f"{r['median_ms']:.3f}" if "median_ms" in r else "-"
        g = f"{r['trtexec_gpu_ms']:.4f}" if r.get("trtexec_gpu_ms") else "-"
        cos = f"{r['cos_sim']:.6f}" if "cos_sim" in r else "-"
        pf = {True: "yes", False: "no"}.get(r.get("python_free_load"), "-")
        note = r.get("error", "")[:60] or (
            f"{r.get('trt_subgraphs')} TRT / {r.get('fallback_subgraphs')} fallback"
            if "trt_subgraphs" in r else f"{r.get('onnx_nodes', '-')} onnx nodes")
        print(f"| {r['variant']} | {r['outcome']} | {ms} | {g} | {pf} | {cos} | {note} |")


if __name__ == "__main__":
    main()
