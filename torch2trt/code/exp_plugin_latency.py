"""Why is `trtexec + plugin` slower than eager on customop (b=32)?

The engine's GPU time (trtexec) is ~0.02 ms, yet the chapter-05 table shows
~0.09 ms through TrtRunner vs ~0.065 ms for eager. This script re-measures
eager / aoti / onnx_trt+plugin / trtexec+plugin and splits the number into
GPU-kernel time and host-side overhead, so the gap gets a cause, not a guess.

Measurements per route:
  event_ms   : common.bench_ms (cuda events on the current stream) -- the
               number the chapter tables use.
  wall_ms    : perf_counter around call + synchronize (host-inclusive).
GPU kernel time is taken with nsys instead (torch.profiler does not see the
kernels TensorRT launches on its own stream reliably):
  nsys profile -t cuda python3 exp_plugin_latency.py --nsys {eager,aoti,py,cli}
  nsys stats -r cuda_gpu_kern_sum <report>
For the TRT engines, TrtRunner is also taken apart step by step, and two
leaner runners are timed: one with preallocated I/O on the current stream,
one replaying a CUDA graph.

Usage:  python3 exp_plugin_latency.py   (needs artifacts from exp_unsupported_op.py)
"""
import ctypes
import statistics
import time

import torch

import common
import models
import routes

ART = common.ARTIFACTS
ctypes.CDLL(str(ART / "liberfinv_plugin.so"))
import tensorrt as trt  # noqa: E402  (after the plugin is registered)

LOGGER = trt.Logger(trt.Logger.ERROR)


def measure(name, fn):
    ev = common.bench_ms(fn, warmup=50, rep=500)["median_ms"]
    wall = common.bench_wall_ms(fn, warmup=50, rep=500)
    print(f"{name:28s} event={ev:.4f}  wall={wall:.4f}")
    return {"route": name, "event_ms": ev, "wall_ms": wall}


class LeanRunner:
    """Preallocated I/O, current stream, no per-call shape/dtype work."""

    def __init__(self, plan, x, graph=False):
        self.engine = trt.Runtime(LOGGER).deserialize_cuda_engine(plan)
        self.ctx = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        iname = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        oname = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
        self.x = x.clone()
        self.ctx.set_input_shape(iname, tuple(x.shape))
        self.out = torch.empty(tuple(self.ctx.get_tensor_shape(oname)),
                               dtype=torch.half, device="cuda")
        self.ctx.set_tensor_address(iname, self.x.data_ptr())
        self.ctx.set_tensor_address(oname, self.out.data_ptr())
        self.s = torch.cuda.current_stream().cuda_stream
        self.graph = None
        if graph:
            self.ctx.execute_async_v3(self.s)        # first run does lazy init
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
            self.graph = g

    def __call__(self, x):
        self.x.copy_(x)
        if self.graph is not None:
            self.graph.replay()
        else:
            self.ctx.execute_async_v3(self.s)
        return self.out


def trtrunner_breakdown(runner, x, rep=2000):
    """Wall time of each step inside TrtRunner.__call__."""
    ctx, eng = runner.ctx, runner.engine
    iname, oname = runner.in_names[0], runner.out_names[0]
    steps = {k: [] for k in ["contiguous+dtype", "set_input_shape+addr",
                             "get_shape+torch.empty+addr", "execute_async_v3",
                             "stream.synchronize", "total"]}
    for i in range(rep + 100):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        t = x.contiguous(); runner._torch_dtype(iname)
        t1 = time.perf_counter()
        ctx.set_input_shape(iname, tuple(t.shape)); ctx.set_tensor_address(iname, t.data_ptr())
        t2 = time.perf_counter()
        buf = torch.empty(tuple(ctx.get_tensor_shape(oname)), dtype=runner._torch_dtype(oname),
                          device="cuda")
        ctx.set_tensor_address(oname, buf.data_ptr())
        t3 = time.perf_counter()
        ctx.execute_async_v3(runner.stream.cuda_stream)
        t4 = time.perf_counter()
        runner.stream.synchronize()
        t5 = time.perf_counter()
        if i >= 100:
            for k, (a, b) in zip(steps, [(t0, t1), (t1, t2), (t2, t3), (t3, t4), (t4, t5), (t0, t5)]):
                steps[k].append((b - a) * 1000)
    print("  TrtRunner.__call__ breakdown (median wall ms):")
    for k, v in steps.items():
        print(f"    {k:28s} {statistics.median(v):.4f}")


def nsys_target(route, n=1000):
    """Run one route n times so `nsys profile --stats` can attribute kernels."""
    model, (x,) = models.get("customop")
    if route == "eager":
        fn = lambda: model(x)
    elif route == "aoti":
        pkg = torch._inductor.aoti_load_package(str(ART / "customop_b32_aoti.pt2"))
        fn = lambda: pkg(x)
    else:
        r = routes.TrtRunner((ART / f"customop_plugin_{route}.plan").read_bytes(), LOGGER)
        fn = lambda: r(x)
    with torch.no_grad():
        for _ in range(n):
            fn()
    torch.cuda.synchronize()


def main():
    import sys
    if len(sys.argv) > 2 and sys.argv[1] == "--nsys":
        return nsys_target(sys.argv[2])
    common.device_info()
    model, (x,) = models.get("customop")
    m32, in32 = models.get("customop", dtype=torch.float32)
    with torch.no_grad():
        ref = m32(*in32).float().cpu()

    pkg = torch._inductor.aoti_load_package(str(ART / "customop_b32_aoti.pt2"))
    plans = {"onnx_trt+plugin": (ART / "customop_plugin_py.plan").read_bytes(),
             "trtexec+plugin": (ART / "customop_plugin_cli.plan").read_bytes()}

    rows = []
    with torch.no_grad():
        rows.append(measure("eager", lambda: model(x)))
        rows.append(measure("aoti", lambda: pkg(x)))
        for name, plan in plans.items():
            r = routes.TrtRunner(plan, LOGGER)
            common.compare(r(x).float().cpu(), ref, name)
            rows.append(measure(f"{name} (TrtRunner)", lambda: r(x)))
            trtrunner_breakdown(r, x)
            lean = LeanRunner(plan, x)
            rows.append(measure(f"{name} (lean)", lambda: lean(x)))
            g = LeanRunner(plan, x, graph=True)
            common.compare(g(x).float().cpu(), ref, f"{name} cudagraph")
            rows.append(measure(f"{name} (cudagraph)", lambda: g(x)))

    print("\n| route | event ms | wall ms |")
    print("|---|---|---|")
    for r in rows:
        print(f"| {r['route']} | {r['event_ms']:.4f} | {r['wall_ms']:.4f} |")


if __name__ == "__main__":
    main()
