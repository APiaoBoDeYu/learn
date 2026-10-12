"""Experiment: what each route does with an op it does not support.

`NoisyActivation` contains `torch.erfinv`. It is a normal aten op, so eager and
Inductor are fine with it, but it has no ONNX opset mapping and no
Torch-TensorRT converter. That makes it a clean probe of the question you will
hit on your first real model: *what happens when one op isn't supported?*

The answers differ in a way that matters a lot for project planning:

  ONNX->TRT   : with no extra work, fails at export (aten::erfinv has no ONNX
                symbolic). The standard fix is two-sided: register a custom
                ONNX op on the export side (`custom::ErfInv`) and give the TRT
                parser a plugin with the same name on the build side. Both the
                Python builder (onnx_trt) and the trtexec CLI are run with and
                without that fix, plus the in-between state (custom node but no
                plugin) so each of the two failure points is shown separately.
  dynamo_trt  : partitions the graph. Supported subgraphs become TRT engines,
                the unsupported op stays in PyTorch. You get a working model
                immediately, at the cost of engine-boundary overhead.
  torch_compile: no issue at all, Inductor just generates a kernel for it.

The script also measures the *cost* of that partial fallback by counting
subgraphs, so "it works" does not get mistaken for "it is fast". A model split
into many alternating TRT/PyTorch chunks can easily be slower than plain eager,
because every boundary is a synchronization plus a memory round-trip.
"""
import subprocess

import torch

import common
import models
import routes
from exp_custom_plugin import (PLUGIN_SO, build_engine_py, export_custom_node_legacy,
                               load_cpp_plugin, trtexec_gpu_ms, Skip)

ART = common.ARTIFACTS


def build_onnx_trt_plugin(model, inputs, name, **kw):
    """onnx_trt + escape hatches: custom ONNX node, C++ plugin, Python builder."""
    import tensorrt as trt
    onnx_path, plan_path = ART / f"{name}_plugin.onnx", ART / f"{name}_plugin_py.plan"
    export_custom_node_legacy(model, inputs, onnx_path)
    load_cpp_plugin()
    try:
        plan, info = build_engine_py(onnx_path, plan_path)
    except Skip as e:
        raise routes.RouteUnsupported(str(e)) from e
    meta = {"plugin_layers": info["plugin_layers"],
            **trtexec_gpu_ms(plan_path, static_plugin=True)}
    return routes.TrtRunner(plan, trt.Logger(trt.Logger.ERROR)), meta


def _trtexec_custom_node(model, inputs, name, plugin):
    """trtexec on the custom-node ONNX, with or without --staticPlugins.

    No Python touches the build: this is what a CI step / Dockerfile would run.
    """
    import tensorrt as trt
    onnx_path, plan_path = ART / f"{name}_plugin.onnx", ART / f"{name}_plugin_cli.plan"
    export_custom_node_legacy(model, inputs, onnx_path)
    cmd = ["trtexec", f"--onnx={onnx_path}", f"--saveEngine={plan_path}", "--fp16"]
    if plugin:
        cmd.append(f"--staticPlugins={PLUGIN_SO}")
    print("  $ " + " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        out = (proc.stdout + proc.stderr).splitlines()
        err = [l for l in out if "checkFallbackPluginImporter" in l or "Plugin not found" in l]
        raise routes.RouteUnsupported("trtexec failed: " + " ".join(err or out[-2:]))
    if plugin:
        load_cpp_plugin()          # so the in-process TrtRunner can deserialize it
    meta = trtexec_gpu_ms(plan_path, static_plugin=plugin)
    return routes.TrtRunner(plan_path.read_bytes(), trt.Logger(trt.Logger.ERROR)), meta


# Order matters: register_custom_op_symbolic and loading the .so are
# process-global, so the "no extra work" rows must run before the plugin rows.
ROUTES = {
    **routes.ROUTES,
    "trtexec_custom_node_no_plugin":
        lambda m, i, name, **kw: _trtexec_custom_node(m, i, name, plugin=False),
    "onnx_trt+plugin": build_onnx_trt_plugin,
    "trtexec+plugin":
        lambda m, i, name, **kw: _trtexec_custom_node(m, i, name, plugin=True),
}
ORDER = ["eager", "torch_compile", "onnx_trt", "trtexec", "dynamo_trt", "ts_trt",
         "trtexec_custom_node_no_plugin", "onnx_trt+plugin", "trtexec+plugin"]


def main():
    common.device_info()
    model, inputs = models.get("customop")
    with torch.no_grad():
        ref = models.get("customop", dtype=torch.float32)[0](
            *models.get("customop", dtype=torch.float32)[1]).float().cpu()

    print("model contains torch.erfinv -- no ONNX opset entry, no TRT converter\n")

    if not PLUGIN_SO.exists():
        subprocess.check_call(["bash", "plugins/build.sh"])

    rows = []
    for route in ORDER:
        print(f"\n--- {route} ---")
        row = {"route": route}
        try:
            runnable, meta = ROUTES[route](model, inputs, name="customop")
        except routes.RouteUnsupported as e:
            print(f"  BUILD FAILED: {str(e)[:200]}")
            row.update(outcome="build_failed", error=str(e)[:400])
            rows.append(row)
            common.save_result("customop_experiment", row)
            continue

        try:
            timing = common.bench_ms(lambda: runnable(*inputs))
            with torch.no_grad():
                got = runnable(*inputs)
        except RuntimeError as e:
            # ts_trt: builds, then the partial-fallback engine rejects its input.
            msg = next((l for l in str(e).splitlines() if "Expected" in l), str(e)[-200:])
            print(f"  RUNTIME FAILED: {msg}")
            row.update(outcome="runtime_failed", error=msg[:400])
            rows.append(row)
            common.save_result("customop_experiment", row)
            continue
        got = (got[0] if isinstance(got, (tuple, list)) else got).float().cpu()
        acc = common.compare(got, ref, route)
        row.update(outcome="built", **meta, **timing, **acc)
        # For dynamo_trt these two counts are the story: how much of the model
        # actually made it into TensorRT.
        if "trt_subgraphs" in meta:
            print(f"  TRT subgraphs={meta['trt_subgraphs']} "
                  f"pytorch-fallback subgraphs={meta['fallback_subgraphs']}")
        if "python_free_load" in meta:
            print(f"  plugin layers={meta.get('plugin_layers', '-')} "
                  f"trtexec --loadEngine ok={meta['python_free_load']} "
                  f"GPU={meta['trtexec_gpu_ms']}ms")
        print(f"  median={timing['median_ms']:.3f}ms")
        rows.append(row)
        common.save_result("customop_experiment", row)

    print("\n\n### unsupported-op verdict\n")
    print("| route | outcome | median ms | trtexec GPU ms | cos | TRT subgraphs | fallback subgraphs |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        ms = f"{r['median_ms']:.3f}" if "median_ms" in r else "-"
        g = f"{r['trtexec_gpu_ms']:.4f}" if r.get("trtexec_gpu_ms") else "-"
        cos = f"{r['cos_sim']:.6f}" if "cos_sim" in r else "-"
        print(f"| {r['route']} | {r['outcome']} | {ms} | {g} | {cos} | "
              f"{r.get('trt_subgraphs','-')} | {r.get('fallback_subgraphs','-')} |")


if __name__ == "__main__":
    main()
