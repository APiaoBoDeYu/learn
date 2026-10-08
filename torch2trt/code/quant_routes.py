"""Quantization routes: INT8 and FP8, via three different mechanisms.

The tutorial's chapters 1-7 were all FP16 and explicitly flagged quantization as
the biggest gap -- and as the one area where TensorRT has a real advantage over
Inductor. This module closes that gap.

Three mechanisms, because they are genuinely different workflows and the
difference matters more than the precision label:

  PTQ via TensorRT's own calibrator (`trt_int8_calib`)
      You hand TensorRT an FP32 ONNX file plus a calibration data feeder, and
      TensorRT decides the per-tensor scales itself. No Q/DQ nodes in the graph.
      This is the "classic" INT8 path and needs no extra Python packages.

  PTQ via modelopt Q/DQ insertion (`modelopt_int8`, `modelopt_fp8`)
      modelopt runs calibration in PyTorch, then *rewrites the model* to contain
      explicit quantize/dequantize ops. The scales are baked into the graph, so
      any downstream compiler (TRT, ONNX Runtime, even Inductor) sees them. This
      is the modern, portable path, and the only one that gives you FP8.

  Weight-only INT8 (`modelopt_int8_wo`)
      Quantizes weights but leaves activations in FP16. Halves weight memory with
      much smaller accuracy risk, but on a compute-bound conv net it usually buys
      little latency -- it is a memory-bandwidth optimization. Included so the
      "why didn't INT8 make it faster" question has a measured answer.

A note on what "INT8 speedup" even means here: quantization helps when the model
is bound by memory bandwidth or by tensor-core throughput at lower precision. It
does NOT help a model that is already launch-latency-bound, and it costs extra
reformat kernels at every precision boundary. Chapter 8 measures both effects.
"""
import time
from pathlib import Path

import torch

import common
import routes
from routes import RouteUnsupported, TrtRunner, _timed, export_onnx

ARTIFACTS = common.ARTIFACTS


# --------------------------------------------------------------------------
# Mechanism 1: TensorRT's native INT8 calibrator
# --------------------------------------------------------------------------
def _make_calibrator(calib_data, cache_path):
    """Build an IInt8EntropyCalibrator2 that feeds `calib_data` to TensorRT.

    TensorRT drives this by callback: it repeatedly calls get_batch() until we
    return None, running the network in FP32 each time to collect activation
    histograms, then picks per-tensor scales that minimise information loss
    (that is the "entropy" in the class name).

    The awkward part of this API, and the reason this helper exists: get_batch()
    must return raw device pointers, and TensorRT does NOT keep the underlying
    memory alive. If the tensor is garbage-collected after we return the pointer,
    TensorRT calibrates on freed memory -- silently, producing plausible-looking
    but wrong scales. Hence `self._current`.
    """
    import tensorrt as trt

    class Calibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self):
            super().__init__()
            self.data = list(calib_data)
            self.idx = 0
            self._current = None
            self.cache_path = Path(cache_path)

        def get_batch_size(self):
            return self.data[0].shape[0]

        def get_batch(self, names):
            if self.idx >= len(self.data):
                return None
            # TensorRT calibrates in FP32.
            self._current = self.data[self.idx].float().contiguous()
            self.idx += 1
            return [int(self._current.data_ptr())]

        def read_calibration_cache(self):
            # Returning a cache lets you skip recalibration on rebuild. We only
            # read it if it exists; chapter 8 measures the build-time saving.
            if self.cache_path.exists():
                return self.cache_path.read_bytes()
            return None

        def write_calibration_cache(self, cache):
            self.cache_path.write_bytes(cache)

    return Calibrator()


def build_trt_int8_calib(model, inputs, name="model", calib_data=None,
                         workspace_gb=4, **kw):
    """ONNX -> TensorRT with INT8 enabled and TensorRT's own entropy calibrator.

    Note FP16 stays enabled alongside INT8. That is deliberate and is what you
    want in practice: the flags are permissions, not mandates, so TensorRT picks
    INT8 where it pays off and FP16 elsewhere. Forcing INT8 everywhere
    (OBEY_PRECISION_CONSTRAINTS) is usually both slower and less accurate.
    """
    import tensorrt as trt

    if calib_data is None:
        raise RouteUnsupported("trt_int8_calib requires calib_data")

    onnx_path = ARTIFACTS / f"{name}_q.onnx"
    engine_path = ARTIFACTS / f"{name}_int8.plan"
    cache_path = ARTIFACTS / f"{name}_calib.cache"

    onnx_s = 0.0
    if not onnx_path.exists():
        # tuple, not list: torch.onnx.export treats a list as "one argument that
        # happens to be a list" and then conv2d gets handed a list instead of a
        # tensor. The error surfaces deep inside conv2d and does not mention
        # export at all, which makes it a genuinely annoying half hour.
        onnx_s = export_onnx(model.float(),
                             tuple(i.float() for i in inputs), onnx_path)

    logger = trt.Logger(trt.Logger.ERROR)
    calibrator = _make_calibrator(calib_data, cache_path)

    def do_build():
        builder = trt.Builder(logger)
        network = builder.create_network()
        parser = trt.OnnxParser(network, logger)
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
                raise RouteUnsupported("onnx parse failed: " + "; ".join(errs))
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calibrator
        plan = builder.build_serialized_network(network, config)
        if plan is None:
            raise RouteUnsupported("TensorRT builder returned no engine")
        engine_path.write_bytes(bytes(plan))
        return bytes(plan)

    try:
        plan, build_s = _timed(do_build)
    except RouteUnsupported:
        raise
    except Exception as e:
        raise RouteUnsupported(f"int8 build failed: {type(e).__name__}: {e}") from e

    return TrtRunner(plan, logger), {
        "build_s": build_s, "onnx_export_s": onnx_s,
        "engine_mb": common.engine_size_mb(engine_path),
        "precision": "int8+fp16", "mechanism": "trt_entropy_calibrator",
    }


# --------------------------------------------------------------------------
# Mechanism 2 & 3: modelopt Q/DQ insertion, then compile through Torch-TensorRT
# --------------------------------------------------------------------------
_MODELOPT_CFGS = {
    "int8": "INT8_DEFAULT_CFG",
    "int8_wo": "INT8_WEIGHT_ONLY_CFG",
    "fp8": "FP8_DEFAULT_CFG",
}


def _modelopt_quantize(model, calib_data, mode: str):
    """Calibrate + rewrite the model with Q/DQ nodes. Returns the new model.

    modelopt's `quantize()` takes a forward_loop callback rather than a data
    loader, so it can drive whatever calling convention your model has. It runs
    the loop once with quantizers in "collect statistics" mode, then freezes the
    scales.
    """
    try:
        import modelopt.torch.quantization as mtq
    except ImportError as e:
        raise RouteUnsupported(f"modelopt not installed: {e}") from e

    cfg_name = _MODELOPT_CFGS[mode]
    cfg = getattr(mtq, cfg_name)

    # modelopt calibration runs in the model's own dtype. We calibrate in fp32
    # for numerical headroom while collecting histograms, then cast back.
    was_half = next(model.parameters()).dtype == torch.half
    model = model.float()

    def forward_loop(m):
        with torch.no_grad():
            for x in calib_data:
                m(x.float())

    try:
        model = mtq.quantize(model, cfg, forward_loop)
    except Exception as e:
        raise RouteUnsupported(
            f"modelopt quantize({mode}) failed: {type(e).__name__}: {e}") from e

    return model.half() if was_half else model, cfg_name


def build_modelopt_trt(model, inputs, name="model", calib_data=None,
                       mode: str = "int8", workspace_gb=4, **kw):
    """modelopt PTQ -> Q/DQ ONNX -> TensorRT builder.

    Note this goes through ONNX, not through torch.export, even though the rest
    of the tutorial prefers the export route. That is not a style choice -- the
    export route does not work here, and *why* it fails is instructive enough
    that chapter 8 spends a section on it:

      * `torch.export(..., strict=False)` succeeds, but Torch-TensorRT then dies
        with "Cannot access data pointer of Tensor (FakeTensor)" -- modelopt's
        fake-quant op calls into a custom CUDA kernel that cannot be traced.
      * On a half-precision Q/DQ model export fails even earlier, with
        `GuardOnDataDependentSymNode` on `Eq(u0, 1)` -- because modelopt's
        quantizer itself contains `if amax.min() < 0`, the exact data-dependent
        control flow pattern chapter 5 is about. The tooling hits the same wall
        it warns users about.

    So the working path is the one NVIDIA documents: let the legacy (trace-based)
    ONNX exporter serialise the Q/DQ nodes, and let the TensorRT builder fold
    them into INT8/FP8 layers. TensorRT needs no calibrator here -- the scales
    are already in the graph, which is the whole point of Q/DQ.
    """
    import tensorrt as trt

    if calib_data is None:
        raise RouteUnsupported(f"modelopt_{mode} requires calib_data")

    t0 = time.perf_counter()
    qmodel, cfg_name = _modelopt_quantize(model, calib_data, mode)
    calib_s = time.perf_counter() - t0

    onnx_path = ARTIFACTS / f"{name}_{mode}_qdq.onnx"
    engine_path = ARTIFACTS / f"{name}_{mode}.plan"

    x = [i.float() for i in inputs]
    try:
        # dynamo=False on purpose: the dynamo exporter cannot translate
        # modelopt's fake-quant ops to ONNX (ConversionError at step 3/3).
        _, onnx_s = _timed(lambda: torch.onnx.export(
            qmodel.float(), tuple(x), str(onnx_path), opset_version=17,
            dynamo=False, input_names=["in0"], output_names=["out"]))
    except Exception as e:
        raise RouteUnsupported(
            f"modelopt_{mode} onnx export failed: {type(e).__name__}: {e}") from e

    n_qdq = _count_qdq(onnx_path)

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
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
        config.set_flag(trt.BuilderFlag.FP16)
        if mode == "fp8":
            config.set_flag(trt.BuilderFlag.FP8)
        else:
            config.set_flag(trt.BuilderFlag.INT8)
        plan = builder.build_serialized_network(network, config)
        if plan is None:
            raise RouteUnsupported("TensorRT builder returned no engine")
        engine_path.write_bytes(bytes(plan))
        return bytes(plan)

    try:
        plan, build_s = _timed(do_build)
    except RouteUnsupported:
        raise
    except Exception as e:
        raise RouteUnsupported(
            f"modelopt_{mode} build failed: {type(e).__name__}: {e}") from e

    return TrtRunner(plan, logger), {
        "build_s": build_s + calib_s + onnx_s, "calib_s": calib_s,
        "onnx_export_s": onnx_s, "qdq_nodes": n_qdq,
        "engine_mb": common.engine_size_mb(engine_path),
        "precision": mode, "mechanism": f"modelopt/{cfg_name}+onnx",
    }


def _count_qdq(onnx_path) -> int:
    """How many QuantizeLinear/DequantizeLinear nodes made it into the ONNX.

    Worth checking for the same reason chapter 4 counts TRT subgraphs: it turns
    "did quantization actually apply" from an assumption into a number. A Q/DQ
    count of 0 means you built a plain FP16 engine and the INT8 flag did nothing.
    """
    import onnx
    g = onnx.load(str(onnx_path))
    return sum(1 for n in g.graph.node if "Quantize" in n.op_type)


def build_modelopt_eager(model, inputs, calib_data=None, mode: str = "int8", **kw):
    """modelopt PTQ with NO compilation -- run the Q/DQ model in eager PyTorch.

    This row exists to separate two effects that are easy to conflate:
    quantization's effect on *accuracy* versus its effect on *latency*. Eager
    Q/DQ is normally SLOWER than fp16 (it adds explicit quantize/dequantize ops
    without any kernel able to exploit them), so if the compiled INT8 route is
    also slow you can tell whether the problem is the quantization or the
    compilation. It is the accuracy reference for the quantized model.
    """
    if calib_data is None:
        raise RouteUnsupported(f"modelopt_{mode}_eager requires calib_data")

    t0 = time.perf_counter()
    qmodel, cfg_name = _modelopt_quantize(model, calib_data, mode)
    calib_s = time.perf_counter() - t0

    @torch.no_grad()
    def run(*args):
        return qmodel(*args)

    return run, {"build_s": calib_s, "calib_s": calib_s,
                 "precision": mode, "mechanism": f"modelopt-eager/{cfg_name}"}


QUANT_ROUTES = {
    "trt_int8_calib": build_trt_int8_calib,
    "modelopt_int8": lambda m, i, **k: build_modelopt_trt(m, i, mode="int8", **k),
    "modelopt_int8_wo": lambda m, i, **k: build_modelopt_trt(m, i, mode="int8_wo", **k),
    "modelopt_fp8": lambda m, i, **k: build_modelopt_trt(m, i, mode="fp8", **k),
    "modelopt_int8_eager": lambda m, i, **k: build_modelopt_eager(m, i, mode="int8", **k),
}

# Make them visible to the generic driver too.
routes.ROUTES.update(QUANT_ROUTES)
