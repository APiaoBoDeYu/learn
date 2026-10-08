"""Benchmark driver: runs the requested (model, route) pairs and tabulates them.

Usage:
    python3 run_bench.py                          # everything (slow: hours)
    python3 run_bench.py --models resnet18        # one model, all routes
    python3 run_bench.py --routes eager,dynamo_trt
    python3 run_bench.py --table                  # re-render from saved results

Design notes worth reading before you trust the numbers:

  * Routes run one at a time in the same process, and we synchronize and sleep
    between them. TensorRT engine builds saturate the GPU; timing route B while
    route A is still building gives nonsense. This is not hypothetical -- during
    development a resnet18 build that normally takes ~20s ran past 40 minutes
    because it was sharing the GPU with another process.
  * Correctness is always measured against eager fp32, not against eager fp16.
    Otherwise an fp16 route gets credit for matching another fp16 route's error.
  * Every row is written to artifacts/results.jsonl as soon as it is produced,
    so a route that hangs or OOMs does not cost you the rows already collected.
  * build_s is wall-clock and includes ONNX export where applicable, because
    that is what you actually wait for in a CI pipeline.
"""
import argparse
import gc
import time
import traceback

import torch

import common
import models
import routes

DEFAULT_MODELS = ["resnet18", "bert_tiny", "ctrlflow", "customop"]
DEFAULT_ROUTES = ["eager", "torch_compile", "onnx_trt", "trtexec",
                  "dynamo_trt", "ts_trt", "aoti"]


def reference_output(name: str):
    """Run the model in fp32 eager to get the ground-truth output.

    fp32 eager is the only thing in this tutorial we treat as "correct". Each
    route's error is measured against this, so fp16 routes are all judged by the
    same yardstick.
    """
    model, inputs = models.get(name, dtype=torch.float32)
    with torch.no_grad():
        return model(*inputs).float().cpu()


def run_one(model_name: str, route_name: str) -> dict:
    """Build + benchmark + check one (model, route) pair."""
    print(f"\n{'='*68}\n{model_name} / {route_name}\n{'='*68}")
    record = {"model": model_name, "route": route_name}

    model, inputs = models.get(model_name)
    build = routes.ROUTES[route_name]

    try:
        runnable, meta = build(model, inputs, name=model_name)
    except routes.RouteUnsupported as e:
        print(f"  UNSUPPORTED: {e}")
        record.update(status="unsupported", error=str(e)[:400])
        common.save_result("bench", record)
        return record
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()
        record.update(status="error", error=f"{type(e).__name__}: {e}"[:400])
        common.save_result("bench", record)
        return record

    record.update(status="ok", **meta)
    print(f"  built in {meta['build_s']:.1f}s")

    # --- latency -------------------------------------------------------
    # A route can build successfully and still fail at inference time -- the
    # TorchScript frontend does exactly this on a partially-fallen-back model
    # (it hands a float tensor to an engine expecting half). That is a result
    # worth recording, not a reason to lose the whole benchmark run, so the
    # inference path gets its own guard.
    try:
        timing = common.bench_ms(lambda: runnable(*inputs))
    except Exception as e:
        print(f"  RUNTIME FAILURE: {type(e).__name__}: {e}")
        record.update(status="runtime_error",
                      error=f"{type(e).__name__}: {e}"[:400])
        common.save_result("bench", record)
        return record
    record.update(timing)
    print(f"  median={timing['median_ms']:.3f}ms p90={timing['p90_ms']:.3f}ms")

    # --- correctness vs fp32 eager -------------------------------------
    try:
        ref = reference_output(model_name)
        with torch.no_grad():
            got = runnable(*inputs)
        got = (got[0] if isinstance(got, (tuple, list)) else got).float().cpu()
        record.update(common.compare(got, ref, f"{model_name}/{route_name}"))
    except Exception as e:
        print(f"  correctness check skipped: {type(e).__name__}: {e}")
        record["accuracy_error"] = str(e)[:200]

    common.save_result("bench", record)

    # Let the GPU settle before the next route builds.
    del runnable
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    return record


def render_table(rows=None) -> None:
    """Print the comparison table from saved results (latest row per pair)."""
    rows = rows if rows is not None else common.load_results()
    latest = {}
    for r in rows:
        if r.get("tag") == "bench" and "model" in r:
            latest[(r["model"], r["route"])] = r
    if not latest:
        print("no results yet -- run the benchmark first")
        return

    model_names = sorted({k[0] for k in latest}, key=lambda m: DEFAULT_MODELS.index(m)
                         if m in DEFAULT_MODELS else 99)
    route_names = sorted({k[1] for k in latest}, key=lambda r: DEFAULT_ROUTES.index(r)
                         if r in DEFAULT_ROUTES else 99)

    for metric, label, fmt in [("median_ms", "median latency (ms)", "{:.3f}"),
                               ("build_s", "build time (s)", "{:.1f}"),
                               ("cos_sim", "cosine sim vs fp32", "{:.5f}")]:
        print(f"\n### {label}\n")
        header = f"| route | " + " | ".join(model_names) + " |"
        print(header)
        print("|" + "---|" * (len(model_names) + 1))
        for route in route_names:
            cells = []
            for m in model_names:
                r = latest.get((m, route))
                if r is None:
                    cells.append("-")
                elif r.get("status") != "ok":
                    cells.append("X")
                elif r.get(metric) is None:
                    cells.append("?")
                else:
                    cells.append(fmt.format(r[metric]))
            print(f"| {route} | " + " | ".join(cells) + " |")

    # speedup vs eager, computed from the same table
    print("\n### speedup vs eager (median latency)\n")
    print("| route | " + " | ".join(model_names) + " |")
    print("|" + "---|" * (len(model_names) + 1))
    for route in route_names:
        cells = []
        for m in model_names:
            base = latest.get((m, "eager"))
            r = latest.get((m, route))
            if not base or not r or r.get("status") != "ok" or not r.get("median_ms"):
                cells.append("-" if r is None else "X")
            else:
                cells.append(f"{base['median_ms'] / r['median_ms']:.2f}x")
        print(f"| {route} | " + " | ".join(cells) + " |")

    print("\n### failures\n")
    any_fail = False
    for (m, route), r in sorted(latest.items()):
        if r.get("status") != "ok":
            any_fail = True
            print(f"- **{m} / {route}** ({r['status']}): {r.get('error','')[:200]}")
    if not any_fail:
        print("(none)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--routes", default=",".join(DEFAULT_ROUTES))
    ap.add_argument("--table", action="store_true",
                    help="only re-render the table from saved results")
    args = ap.parse_args()

    if args.table:
        render_table()
        return

    common.device_info()
    for m in args.models.split(","):
        for r in args.routes.split(","):
            run_one(m.strip(), r.strip())

    print("\n\n" + "#" * 68)
    render_table()


if __name__ == "__main__":
    main()
