"""Follow-up experiment: is INT8's failure to beat FP16 a batch-size artifact?

The main quantization experiment found something that contradicts the usual
expectation: at batch=8, no INT8 or FP8 route beat FP16 on this GPU (best INT8
was 0.98x of FP16 TRT). Before writing that into the tutorial as a conclusion,
it has to be checked, because the most likely innocent explanation is that
batch=8 on resnet18 is too small to saturate the tensor cores -- at that size
latency can be dominated by kernel launch overhead and by the reformat kernels
that INT8 inserts at every precision boundary, both of which are fixed costs
that quantization makes worse rather than better.

If that explanation is right, INT8 should overtake FP16 as batch grows. If INT8
still loses at batch=64, the conclusion is about this hardware/model, not about
the measurement.

This is the difference between "my benchmark says X" and "X is true": one
adversarial follow-up on the most likely confound.
"""
import gc
import time

import torch

import common
import data
import quant_routes
import routes

BATCHES = [1, 8, 32, 64]
ROUTES = ["onnx_trt", "modelopt_int8", "modelopt_fp8"]


def make_model(dtype=torch.half):
    import torchvision
    return torchvision.models.resnet18(
        weights="IMAGENET1K_V1").eval().to("cuda", dtype)


def main():
    common.device_info()
    if not data.available():
        raise SystemExit(f"need imagenette at {data.DATA_ROOT}")

    results = {}
    for batch in BATCHES:
        # Calibration set is rebuilt per batch size because the calibrator's
        # batch dimension has to match the engine's.
        calib = data.calib_batches(n_batches=max(2, 128 // batch), batch=batch)
        example = calib[0].half()
        print(f"\n{'#'*68}\n# batch={batch}  input={tuple(example.shape)}\n{'#'*68}")

        for route in ROUTES:
            # Each batch size needs its own ONNX/engine; clear the cached files
            # so we do not silently reuse an engine built for another shape.
            for f in common.ARTIFACTS.glob(f"rn18b{batch}*"):
                f.unlink()
            print(f"\n--- {route} @ b={batch} ---")
            model = make_model()
            try:
                runnable, meta = routes.ROUTES[route](
                    model, (example,), name=f"rn18b{batch}", calib_data=calib)
                timing = common.bench_ms(lambda: runnable(example))
            except Exception as e:
                print(f"  FAILED: {type(e).__name__}: {str(e)[:180]}")
                results[(batch, route)] = None
                continue

            # Per-image latency is the fair comparison across batch sizes.
            per_img = timing["median_ms"] / batch
            results[(batch, route)] = {**timing, "per_img_ms": per_img,
                                       "precision": meta.get("precision")}
            print(f"  median={timing['median_ms']:.3f}ms "
                  f"per_image={per_img:.4f}ms")
            common.save_result("quant_batch", {"batch": batch, "route": route,
                                               **results[(batch, route)]})
            del runnable, model
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
            time.sleep(2)

    print("\n\n### INT8/FP8 vs FP16 across batch sizes (resnet18)\n")
    print("| batch | " + " | ".join(f"{r} (ms)" for r in ROUTES) +
          " | int8 vs fp16 | fp8 vs fp16 |")
    print("|" + "---|" * (len(ROUTES) + 3))
    for batch in BATCHES:
        base = results.get((batch, "onnx_trt"))
        cells = []
        for r in ROUTES:
            v = results.get((batch, r))
            cells.append(f"{v['median_ms']:.3f}" if v else "X")
        def ratio(rt):
            v = results.get((batch, rt))
            return f"{base['median_ms']/v['median_ms']:.2f}x" if base and v else "-"
        print(f"| {batch} | " + " | ".join(cells) +
              f" | {ratio('modelopt_int8')} | {ratio('modelopt_fp8')} |")

    print("\n### per-image latency (ms) -- throughput view\n")
    print("| batch | " + " | ".join(ROUTES) + " |")
    print("|" + "---|" * (len(ROUTES) + 1))
    for batch in BATCHES:
        cells = []
        for r in ROUTES:
            v = results.get((batch, r))
            cells.append(f"{v['per_img_ms']:.4f}" if v else "X")
        print(f"| {batch} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
