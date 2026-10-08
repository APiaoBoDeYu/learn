"""Experiment: does INT8/FP8 quantization actually pay off, and what does it cost?

This is the experiment chapters 1-7 could not run (no modelopt installed, and no
real weights/data to measure accuracy loss against). It answers three questions
that FP16-only benchmarking cannot:

  1. Latency: how much faster is INT8 than FP16, really?
  2. Accuracy: how much top-1 do you actually lose? (measured on 800 real
     ImageNet images with ImageNet-pretrained weights -- a random-weight model
     has no accuracy to lose, so this experiment requires real data)
  3. Cost: how much longer does calibration + build take?

Unlike the rest of the tutorial this uses `weights="IMAGENET1K_V1"`, because
"cosine similarity vs fp32" is the wrong metric for quantization. A quantized
model can have visibly lower cosine similarity and identical top-1 -- or the
reverse, which is the dangerous case. Only real accuracy settles it.
"""
import gc
import time
import traceback

import torch

import common
import data
import quant_routes
import routes

# FP16 references first, then the quantized routes. Order matters: the FP16 rows
# are the baseline every quantized row is judged against.
ROUTE_ORDER = [
    ("eager", {}),                  # fp16 eager, accuracy + latency baseline
    ("dynamo_trt", {}),             # fp16 TRT, the bar INT8 has to beat
    ("onnx_trt", {}),               # fp16 TRT via ONNX
    ("modelopt_int8_eager", {}),    # INT8 accuracy without compilation
    ("trt_int8_calib", {}),         # INT8 via TensorRT's own calibrator
    ("modelopt_int8", {}),          # INT8 via modelopt Q/DQ + TRT
    ("modelopt_int8_wo", {}),       # INT8 weights only, FP16 activations
    ("modelopt_fp8", {}),           # FP8 via modelopt Q/DQ + TRT
]


def make_model(dtype=torch.half):
    """ImageNet-pretrained resnet18 -- real weights, so accuracy is meaningful."""
    import torchvision
    return torchvision.models.resnet18(
        weights="IMAGENET1K_V1").eval().to("cuda", dtype)


def main():
    common.device_info()
    if not data.available():
        raise SystemExit(f"need imagenette at {data.DATA_ROOT}; see data.py docstring")

    # One calibration set, shared by every quantized route. If each route got its
    # own sample, a route could score better purely from luckier images.
    print("building shared calibration set (128 real train images)...")
    calib = data.calib_batches(n_batches=16, batch=8)
    example = calib[0].half()
    print(f"  {len(calib)} batches of {tuple(calib[0].shape)}")

    rows = []
    for route_name, extra in ROUTE_ORDER:
        print(f"\n{'='*68}\n{route_name}\n{'='*68}")
        row = {"route": route_name}
        model = make_model()
        build = routes.ROUTES[route_name]
        try:
            runnable, meta = build(model, (example,), name="resnet18q",
                                   calib_data=calib, **extra)
        except routes.RouteUnsupported as e:
            print(f"  UNSUPPORTED: {str(e)[:250]}")
            row.update(status="unsupported", error=str(e)[:400])
            rows.append(row); common.save_result("quant", row); continue
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()
            row.update(status="error", error=f"{type(e).__name__}: {e}"[:400])
            rows.append(row); common.save_result("quant", row); continue

        row.update(status="ok", **meta)
        print(f"  built in {meta['build_s']:.1f}s ({meta.get('mechanism','fp16')})")

        try:
            timing = common.bench_ms(lambda: runnable(example))
        except Exception as e:
            print(f"  RUNTIME FAILURE: {type(e).__name__}: {e}")
            row.update(status="runtime_error", error=str(e)[:400])
            rows.append(row); common.save_result("quant", row); continue
        row.update(timing)
        print(f"  median={timing['median_ms']:.3f}ms p90={timing['p90_ms']:.3f}ms")

        # The measurement that matters: real top-1 on real images.
        try:
            acc = data.top1(runnable, batch=8, limit=800)
            row.update(acc)
            print(f"  top1={acc['top1']*100:.2f}% over {acc['n_eval']} images")
        except Exception as e:
            print(f"  accuracy eval failed: {type(e).__name__}: {e}")
            row["accuracy_error"] = str(e)[:250]

        rows.append(row)
        common.save_result("quant", row)
        del runnable, model
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
        time.sleep(2)

    render(rows)


def render(rows):
    base = next((r for r in rows if r["route"] == "eager"
                 and r.get("status") == "ok"), None)
    fp16_trt = next((r for r in rows if r["route"] == "dynamo_trt"
                     and r.get("status") == "ok"), None)

    print("\n\n### quantization results (resnet18, ImageNet-pretrained, b=8)\n")
    print("| route | precision | median ms | vs fp16 eager | vs fp16 TRT | "
          "top1 | build s |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        if r.get("status") != "ok" or "median_ms" not in r:
            print(f"| {r['route']} | - | X | - | - | - | - |")
            continue
        ms = r["median_ms"]
        sp_e = f"{base['median_ms']/ms:.2f}x" if base else "-"
        sp_t = f"{fp16_trt['median_ms']/ms:.2f}x" if fp16_trt else "-"
        t1 = f"{r['top1']*100:.2f}%" if r.get("top1") is not None else "-"
        print(f"| {r['route']} | {r.get('precision','fp16')} | {ms:.3f} | "
              f"{sp_e} | {sp_t} | {t1} | {r['build_s']:.1f} |")

    if base and base.get("top1") is not None:
        print(f"\nFP16 baseline top-1: **{base['top1']*100:.2f}%**")
        print("\n### accuracy delta vs fp16 eager (percentage points)\n")
        print("| route | top1 | delta |")
        print("|---|---|---|")
        for r in rows:
            if r.get("top1") is None:
                continue
            d = (r["top1"] - base["top1"]) * 100
            print(f"| {r['route']} | {r['top1']*100:.2f}% | {d:+.2f} pp |")

    print("\n### failures\n")
    fails = [r for r in rows if r.get("status") != "ok"]
    if not fails:
        print("(none)")
    for r in fails:
        print(f"- **{r['route']}** ({r['status']}): {r.get('error','')[:220]}")


if __name__ == "__main__":
    main()
