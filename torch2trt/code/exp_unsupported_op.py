"""Experiment: what each route does with an op it does not support.

`NoisyActivation` contains `torch.erfinv`. It is a normal aten op, so eager and
Inductor are fine with it, but it has no ONNX opset mapping and no
Torch-TensorRT converter. That makes it a clean probe of the question you will
hit on your first real model: *what happens when one op isn't supported?*

The answers differ in a way that matters a lot for project planning:

  ONNX->TRT   : fails at export or at parse. You must write a plugin, rewrite
                the op, or give up. All-or-nothing.
  dynamo_trt  : partitions the graph. Supported subgraphs become TRT engines,
                the unsupported op stays in PyTorch. You get a working model
                immediately, at the cost of engine-boundary overhead.
  torch_compile: no issue at all, Inductor just generates a kernel for it.

The script also measures the *cost* of that partial fallback by counting
subgraphs, so "it works" does not get mistaken for "it is fast". A model split
into many alternating TRT/PyTorch chunks can easily be slower than plain eager,
because every boundary is a synchronization plus a memory round-trip.
"""
import torch

import common
import models
import routes


def main():
    common.device_info()
    model, inputs = models.get("customop")
    with torch.no_grad():
        ref = models.get("customop", dtype=torch.float32)[0](
            *models.get("customop", dtype=torch.float32)[1]).float().cpu()

    print("model contains torch.erfinv -- no ONNX opset entry, no TRT converter\n")

    rows = []
    for route in ["eager", "torch_compile", "onnx_trt", "dynamo_trt", "ts_trt"]:
        print(f"\n--- {route} ---")
        row = {"route": route}
        try:
            runnable, meta = routes.ROUTES[route](model, inputs, name="customop")
        except routes.RouteUnsupported as e:
            print(f"  BUILD FAILED: {str(e)[:200]}")
            row.update(outcome="build_failed", error=str(e)[:400])
            rows.append(row)
            common.save_result("customop_experiment", row)
            continue

        timing = common.bench_ms(lambda: runnable(*inputs))
        with torch.no_grad():
            got = runnable(*inputs)
        got = (got[0] if isinstance(got, (tuple, list)) else got).float().cpu()
        acc = common.compare(got, ref, route)
        row.update(outcome="built", **meta, **timing, **acc)
        # For dynamo_trt these two counts are the story: how much of the model
        # actually made it into TensorRT.
        if "trt_subgraphs" in meta:
            print(f"  TRT subgraphs={meta['trt_subgraphs']} "
                  f"pytorch-fallback subgraphs={meta['fallback_subgraphs']}")
        print(f"  median={timing['median_ms']:.3f}ms")
        rows.append(row)
        common.save_result("customop_experiment", row)

    print("\n\n### unsupported-op verdict\n")
    print("| route | outcome | median ms | TRT subgraphs | fallback subgraphs |")
    print("|---|---|---|---|---|")
    for r in rows:
        ms = f"{r['median_ms']:.3f}" if "median_ms" in r else "-"
        print(f"| {r['route']} | {r['outcome']} | {ms} | "
              f"{r.get('trt_subgraphs','-')} | {r.get('fallback_subgraphs','-')} |")


if __name__ == "__main__":
    main()
