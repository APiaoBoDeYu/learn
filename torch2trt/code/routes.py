"""The conversion routes, all behind one uniform interface.

Every route is a function `build(model, inputs, **kw) -> (runnable, meta)`:
  runnable : a callable with the same signature as the original model
  meta     : dict with build_s, and whatever route-specific facts are worth
             recording (engine size, number of TRT subgraphs, fallback ops...)

Keeping them uniform is what makes chapter 06's table trustworthy: the only
thing that varies between rows is the route, not the timing code or the input
data. A route that cannot handle a given model raises `RouteUnsupported`, which
the driver records as a real result (an X in the table) rather than a crash.

The six routes (chapter 10 adds custom-op / plugin variants of R2-R4 in
exp_custom_plugin.py), and why each one exists in this tutorial:

  R0 eager          - the baseline. Without it, every speedup number is
                      meaningless.
  R1 torch.compile  - no TensorRT at all; Inductor generates Triton kernels.
                      The honest question "do I even need TensorRT?" deserves a
                      measured answer.
  R2 onnx_trt       - the user's current route: torch.onnx.export -> ONNX file
                      -> TensorRT builder -> engine. Deployment-friendly
                      (Python-free runtime) but two format hops from PyTorch.
  R3 trtexec        - same ONNX file, but built by the trtexec CLI instead of
                      the Python API. Same engine class, totally different
                      workflow ergonomics.
  R4 dynamo_trt     - Torch-TensorRT via torch.export. One hop instead of two,
                      keeps PyTorch semantics, supports per-subgraph fallback.
  R5 ts_trt         - Torch-TensorRT via TorchScript. The legacy route; in
                      maintenance mode upstream. Included because a lot of
                      existing production code is still on it.
  R6 aoti           - torch.export -> AOTInductor .so. Not TensorRT at all, but
                      it is the other "compile to a deployable artifact with no
                      Python" answer, so it belongs in the comparison.
"""
import os
import time
from pathlib import Path

import torch

import common

ARTIFACTS = common.ARTIFACTS


class RouteUnsupported(Exception):
    """Raised when a route genuinely cannot handle a model.

    This is a *result*, not a bug: "ONNX export refuses this model" is exactly
    the kind of fact this tutorial exists to measure, so the driver catches it
    and records it.
    """


def _timed(fn):
    t0 = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - t0


# --------------------------------------------------------------------------
# R0: eager
# --------------------------------------------------------------------------
def build_eager(model, inputs, **kw):
    """No conversion at all. Baseline for every speedup ratio."""
    @torch.no_grad()
    def run(*args):
        return model(*args)
    return run, {"build_s": 0.0}


# --------------------------------------------------------------------------
# R1: torch.compile (Inductor / Triton) -- no TensorRT involved
# --------------------------------------------------------------------------
def build_torch_compile(model, inputs, mode: str = "max-autotune-no-cudagraphs", **kw):
    """Compile with Inductor. Warms up once so build_s includes the JIT cost.

    We use max-autotune-no-cudagraphs rather than plain max-autotune because
    cudagraphs would also change the *launch* behaviour, and we want this row to
    measure Inductor's codegen, not cudagraph capture. Chapter 04 measures
    cudagraphs separately so the two effects stay distinguishable.
    """
    compiled = torch.compile(model, mode=mode, dynamic=False)

    def warm():
        with torch.no_grad():
            compiled(*inputs)
        torch.cuda.synchronize()

    _, build_s = _timed(warm)

    @torch.no_grad()
    def run(*args):
        return compiled(*args)
    return run, {"build_s": build_s, "mode": mode}


# --------------------------------------------------------------------------
# R2: torch -> ONNX -> TensorRT (Python API)
# --------------------------------------------------------------------------
def export_onnx(model, inputs, path, dynamic_axes=None, opset: int = 17,
                dynamo: bool = False) -> float:
    """Export to ONNX, returning elapsed seconds. Raises RouteUnsupported on failure."""
    try:
        _, secs = _timed(lambda: torch.onnx.export(
            model, inputs, str(path),
            opset_version=opset,
            input_names=[f"in{i}" for i in range(len(inputs))],
            output_names=["out"],
            dynamic_axes=dynamic_axes,
            dynamo=dynamo,
        ))
        return secs
    except Exception as e:
        raise RouteUnsupported(f"onnx export failed: {type(e).__name__}: {e}") from e


def build_onnx_trt(model, inputs, name: str = "model", fp16: bool = True,
                   workspace_gb: int = 4, profile=None, **kw):
    """The classic two-hop route: ONNX file, then the TensorRT Python builder.

    `profile` is an optional list of (input_name, min, opt, max) shape tuples;
    supplying it is what makes the engine accept dynamic shapes. Chapter 03
    shows what happens when you forget it.
    """
    import tensorrt as trt

    onnx_path = ARTIFACTS / f"{name}.onnx"
    engine_path = ARTIFACTS / f"{name}_py.plan"
    onnx_s = export_onnx(model, inputs, onnx_path,
                         dynamic_axes=kw.get("dynamic_axes"))

    logger = trt.Logger(trt.Logger.ERROR)

    def do_build():
        builder = trt.Builder(logger)
        network = builder.create_network()
        parser = trt.OnnxParser(network, logger)
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
                raise RouteUnsupported("onnx parse failed: " + "; ".join(errs))
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                     workspace_gb * 1 << 30)
        if fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        if profile:
            p = builder.create_optimization_profile()
            for iname, mn, opt, mx in profile:
                p.set_shape(iname, mn, opt, mx)
            config.add_optimization_profile(p)
        plan = builder.build_serialized_network(network, config)
        if plan is None:
            raise RouteUnsupported("TensorRT builder returned no engine")
        with open(engine_path, "wb") as f:
            f.write(plan)
        return bytes(plan)

    plan, build_s = _timed(do_build)
    runnable = TrtRunner(plan, logger)
    return runnable, {
        "build_s": build_s,
        "onnx_export_s": onnx_s,
        "engine_mb": common.engine_size_mb(engine_path),
        "onnx_mb": common.engine_size_mb(onnx_path),
    }


class TrtRunner:
    """Minimal TensorRT execution wrapper.

    Written out rather than pulled from a helper library on purpose: this is the
    code you are signing up to own when you choose the ONNX->TRT route, and its
    ~40 lines (bindings, device buffers, stream, shape setting) are a real part
    of that route's cost. Compare against the dynamo route, where the runnable
    you get back is already a torch Module.
    """

    def __init__(self, plan: bytes, logger):
        import tensorrt as trt
        self.trt = trt
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(plan)
        self.ctx = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self.in_names = [n for n in self._names()
                         if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.out_names = [n for n in self._names()
                          if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]

    def _names(self):
        return [self.engine.get_tensor_name(i)
                for i in range(self.engine.num_io_tensors)]

    def _torch_dtype(self, name):
        import tensorrt as trt
        return {
            trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16,
            trt.DataType.INT32: torch.int32, trt.DataType.INT64: torch.int64,
            trt.DataType.BOOL: torch.bool, trt.DataType.INT8: torch.int8,
        }[self.engine.get_tensor_dtype(name)]

    def __call__(self, *args):
        for name, tensor in zip(self.in_names, args):
            tensor = tensor.contiguous()
            # The engine may have been built expecting int32 ids even though
            # torch hands us int64; casting here rather than failing is what
            # real serving code ends up doing.
            want = self._torch_dtype(name)
            if tensor.dtype != want:
                tensor = tensor.to(want)
            self.ctx.set_input_shape(name, tuple(tensor.shape))
            self.ctx.set_tensor_address(name, tensor.data_ptr())
            self._keep = tensor  # keep the cast copy alive until execute
        outs = []
        for name in self.out_names:
            shape = tuple(self.ctx.get_tensor_shape(name))
            buf = torch.empty(shape, dtype=self._torch_dtype(name), device="cuda")
            self.ctx.set_tensor_address(name, buf.data_ptr())
            outs.append(buf)
        self.ctx.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return outs[0] if len(outs) == 1 else tuple(outs)


# --------------------------------------------------------------------------
# R3: trtexec CLI on the same ONNX file
# --------------------------------------------------------------------------
def build_trtexec(model, inputs, name: str = "model", fp16: bool = True,
                  extra_args=None, **kw):
    """Build the engine with the trtexec CLI, then run it through TrtRunner.

    Same engine class as R2, but the build is a subprocess with no Python in the
    loop. That matters in practice: it is the route you use in a Dockerfile or
    CI step, and it gives you trtexec's own profiling output for free.
    """
    import subprocess
    import tensorrt as trt

    onnx_path = ARTIFACTS / f"{name}.onnx"
    engine_path = ARTIFACTS / f"{name}_cli.plan"
    onnx_s = 0.0
    if not onnx_path.exists():
        onnx_s = export_onnx(model, inputs, onnx_path,
                             dynamic_axes=kw.get("dynamic_axes"))

    cmd = ["trtexec", f"--onnx={onnx_path}", f"--saveEngine={engine_path}"]
    if fp16:
        cmd.append("--fp16")
    cmd += list(extra_args or [])

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    build_s = time.perf_counter() - t0
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-5:])
        raise RouteUnsupported(f"trtexec failed (rc={proc.returncode}): {tail}")

    with open(engine_path, "rb") as f:
        plan = f.read()
    runnable = TrtRunner(plan, trt.Logger(trt.Logger.ERROR))
    return runnable, {
        "build_s": build_s,
        "onnx_export_s": onnx_s,
        "engine_mb": common.engine_size_mb(engine_path),
        "cmd": " ".join(cmd),
    }


# --------------------------------------------------------------------------
# R4: Torch-TensorRT, dynamo / torch.export frontend
# --------------------------------------------------------------------------
def build_dynamo_trt(model, inputs, fp16: bool = True, min_block_size: int = 1,
                     dynamic_shapes=None, **kw):
    """The one-hop route: torch.export -> Torch-TensorRT -> GraphModule.

    Two things to notice versus R2/R3:
      - There is no ONNX file. The graph comes from torch.export, so PyTorch
        semantics are preserved rather than round-tripped through another IR.
      - Unsupported ops do not fail the build. Torch-TensorRT partitions the
        graph and leaves those subgraphs in PyTorch, so you get a working
        (if slower) model. That is what chapter 05 measures on `customop`.
    """
    try:
        import torch_tensorrt as tt
    except ImportError as e:
        raise RouteUnsupported(f"torch_tensorrt not installed: {e}") from e

    precisions = {torch.half, torch.float} if fp16 else {torch.float}

    # offload_module_to_cpu=True (needed on Jetson's unified memory, see ch.1)
    # MOVES THE CALLER'S MODEL TO THE CPU AND LEAVES IT THERE. Torch-TensorRT
    # does this in place and never restores it, so every later use of `model` --
    # a subsequent route, an accuracy reference, anything -- fails with
    # "Expected all tensors to be on the same device". Record the original
    # devices and put them back in a finally block.
    #
    # This is not a theoretical hazard: it silently broke three routes in the
    # dynamic-shape experiment, and the symptom (an ONNX export failing on
    # device mismatch) points nowhere near the actual cause.
    _orig_devices = {n: p.device for n, p in
                     list(model.named_parameters()) + list(model.named_buffers())}

    def _restore():
        for n, p in list(model.named_parameters()) + list(model.named_buffers()):
            want = _orig_devices.get(n)
            if want is not None and p.device != want:
                p.data = p.data.to(want)

    def do_build():
        ep = torch.export.export(model, inputs, dynamic_shapes=dynamic_shapes)
        return tt.dynamo.compile(
            ep, inputs=list(inputs),
            enabled_precisions=precisions,
            min_block_size=min_block_size,
            truncate_double=True,
            # Thor has unified memory; the builder warns about headroom unless
            # we let it stage weights on the host during engine construction.
            offload_module_to_cpu=True,
        )

    try:
        gm, build_s = _timed(do_build)
    except Exception as e:
        raise RouteUnsupported(f"dynamo_trt build failed: {type(e).__name__}: {e}") from e
    finally:
        _restore()

    n_trt = sum(1 for n in gm.graph.nodes if "_run_on_acc" in str(n.target))
    n_fallback = sum(1 for n in gm.graph.nodes if "_run_on_gpu" in str(n.target))

    @torch.no_grad()
    def run(*args):
        return gm(*args)
    return run, {"build_s": build_s, "trt_subgraphs": n_trt,
                 "fallback_subgraphs": n_fallback}


# --------------------------------------------------------------------------
# R5: Torch-TensorRT, TorchScript frontend (legacy / maintenance mode)
# --------------------------------------------------------------------------
def build_ts_trt(model, inputs, fp16: bool = True, **kw):
    """The legacy route. Kept because plenty of production code still runs it.

    Note the trace step: `torch.jit.trace` is where data-dependent control flow
    gets silently baked in, which is precisely the failure chapter 05 shows.
    """
    try:
        import torch_tensorrt as tt
    except ImportError as e:
        raise RouteUnsupported(f"torch_tensorrt not installed: {e}") from e

    def do_build():
        scripted = torch.jit.trace(model, inputs)
        return tt.ts.compile(
            scripted,
            inputs=[tt.Input(shape=tuple(i.shape), dtype=i.dtype) for i in inputs],
            enabled_precisions={torch.half} if fp16 else {torch.float},
            truncate_long_and_double=True,
        )

    try:
        mod, build_s = _timed(do_build)
    except Exception as e:
        raise RouteUnsupported(f"ts_trt build failed: {type(e).__name__}: {e}") from e

    @torch.no_grad()
    def run(*args):
        return mod(*args)
    return run, {"build_s": build_s}


# --------------------------------------------------------------------------
# R6: torch.export -> AOTInductor .so (no TensorRT)
# --------------------------------------------------------------------------
def build_aoti(model, inputs, name: str = "model", **kw):
    """Compile to a self-contained .so via AOTInductor.

    This is the "deploy without Python, without TensorRT" answer and the modern
    replacement for TorchScript-based deployment. It belongs in this comparison
    because it competes with the TRT routes on the same axis (an AOT artifact
    you ship) while keeping Inductor's codegen instead of TRT's tactics.
    """
    so_path = ARTIFACTS / f"{name}_aoti.so"

    def do_build():
        ep = torch.export.export(model, inputs)
        return torch._inductor.aoti_compile_and_package(
            ep, package_path=str(so_path).replace(".so", ".pt2"))

    try:
        pkg, build_s = _timed(do_build)
        loaded = torch._inductor.aoti_load_package(pkg)
    except Exception as e:
        raise RouteUnsupported(f"aoti build failed: {type(e).__name__}: {e}") from e

    @torch.no_grad()
    def run(*args):
        return loaded(*args)
    return run, {"build_s": build_s, "artifact_mb": common.engine_size_mb(pkg)}


ROUTES = {
    "eager": build_eager,
    "torch_compile": build_torch_compile,
    "onnx_trt": build_onnx_trt,
    "trtexec": build_trtexec,
    "dynamo_trt": build_dynamo_trt,
    "ts_trt": build_ts_trt,
    "aoti": build_aoti,
}
