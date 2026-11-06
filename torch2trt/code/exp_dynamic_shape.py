"""Experiment: dynamic shapes across the routes.

Chapters 1-7 were all static-shape, and chapter 6 flagged this as a gap. It
matters because static shapes are often a lie in production: request batch size
varies with traffic, and sequence length varies per request. The workarounds
(pad everything to max, or build one engine per shape) both cost real money, so
how well each route handles genuine dynamism is a first-class selection
criterion.

The routes express dynamism in three completely different ways, and that
difference IS the chapter:

  ONNX -> TRT   : `dynamic_axes` at export, THEN an optimization profile
                  (min/opt/max) at build. Two places, and forgetting either one
                  silently gives you a static engine.
  torch.export  : `torch.export.Dim` objects with explicit bounds, checked
                  symbolically at export time -- it will reject shapes your model
                  cannot actually handle.
  torch.compile : nothing. `dynamic=True` and it specialises lazily, recompiling
                  when it sees a shape that breaks a guard.

Four things get measured:

  1. Does each route accept a range of shapes at all?
  2. What does it cost at the shape the engine was tuned for (`opt`)?
  3. What does it cost AWAY from that shape -- the question that decides whether
     one dynamic engine can replace N static ones?
  4. How does one dynamic engine compare to a purpose-built static engine at the
     same shape? That is the real price of dynamism.
"""
import gc
import time

import torch

import common
import models
import routes

# Sequence lengths to exercise. 128 is the `opt` point every dynamic engine is
# tuned for; 32 and 384 probe how badly performance degrades away from it.
SEQ_LENS = [32, 128, 384]
OPT_SEQ = 128
BATCH = 8


def bert_inputs(seq: int, batch: int = BATCH):
    torch.manual_seed(0)
    return torch.randint(0, 30000, (batch, seq), device="cuda", dtype=torch.long)


def build_static_onnx_trt(model, seq: int):
    """A purpose-built static engine at one sequence length -- the thing a
    dynamic engine has to be compared against."""
    name = f"bert_s{seq}"
    for f in common.ARTIFACTS.glob(f"{name}*"):
        f.unlink()
    return routes.build_onnx_trt(model, (bert_inputs(seq),), name=name)


def build_dynamic_onnx_trt(model, min_s=32, opt_s=OPT_SEQ, max_s=384):
    """ONNX + TRT with dynamic sequence length.

    Both halves are required and they are easy to get half-right:
      * `dynamic_axes` tells the ONNX exporter to emit a symbolic dim.
      * the optimization profile tells TensorRT what range to support and --
        crucially -- which shape to tune tactics for. TensorRT optimises for
        `opt` alone; min/max only define what is legal.
    """
    name = "bert_dyn"
    for f in common.ARTIFACTS.glob(f"{name}*"):
        f.unlink()
    return routes.build_onnx_trt(
        model, (bert_inputs(opt_s),), name=name,
        dynamic_axes={"in0": {1: "seq"}, "out": {0: "batch"}},
        profile=[("in0", (BATCH, min_s), (BATCH, opt_s), (BATCH, max_s))],
    )


def build_dynamic_dynamo_trt(model, min_s=32, opt_s=OPT_SEQ, max_s=384):
    """Torch-TensorRT with a torch.export Dim.

    `Dim("seq", min=.., max=..)` is checked symbolically during export, so if the
    model does something that only works for one length, you find out here rather
    than at inference. Torch-TensorRT turns the Dim's bounds into the TRT profile
    automatically -- one place instead of two.
    """
    seq = torch.export.Dim("seq", min=min_s, max=max_s)
    return routes.build_dynamo_trt(
        model, (bert_inputs(opt_s),), dynamic_shapes={"ids": {1: seq}})


def build_dynamic_torch_compile(model):
    """torch.compile with dynamic=True -- no shape declaration at all."""
    compiled = torch.compile(model, dynamic=True)
    t0 = time.perf_counter()
    with torch.no_grad():
        compiled(bert_inputs(OPT_SEQ))
    torch.cuda.synchronize()

    @torch.no_grad()
    def run(*a):
        return compiled(*a)
    return run, {"build_s": time.perf_counter() - t0}


def probe(label, runnable, model):
    """Time a runnable at every sequence length; record failures per length."""
    print(f"\n--- {label} ---")
    out = {}
    for seq in SEQ_LENS:
        x = bert_inputs(seq)
        try:
            with torch.no_grad():
                ref = model(x).float().cpu()
                got = runnable(x)
            got = (got[0] if isinstance(got, (tuple, list)) else got).float().cpu()
            cos = torch.nn.functional.cosine_similarity(
                got.flatten(), ref.flatten(), dim=0).item()
            t = common.bench_ms(lambda: runnable(x), warmup=10, rep=50)
            out[seq] = {"median_ms": t["median_ms"], "cos": cos}
            print(f"  seq={seq:4d}: {t['median_ms']:.3f}ms  cos={cos:.5f}")
        except Exception as e:
            out[seq] = None
            print(f"  seq={seq:4d}: FAILED {type(e).__name__}: {str(e)[:110]}")
    return out


def main():
    common.device_info()
    model, _ = models.get("bert_tiny", seq=OPT_SEQ)

    table = {}

    # eager -- the reference: handles any shape, no build step, no constraints.
    @torch.no_grad()
    def eager(*a):
        return model(*a)
    table["eager (fp16)"] = probe("eager", eager, model)

    builders = [
        ("onnx_trt dynamic", lambda: build_dynamic_onnx_trt(model)),
        ("dynamo_trt dynamic", lambda: build_dynamic_dynamo_trt(model)),
        ("torch_compile dynamic", lambda: build_dynamic_torch_compile(model)),
    ]
    for label, fn in builders:
        try:
            runnable, meta = fn()
        except Exception as e:
            print(f"\n--- {label} ---\n  BUILD FAILED: {type(e).__name__}: "
                  f"{str(e)[:220]}")
            table[label] = {s: None for s in SEQ_LENS}
            continue
        print(f"\n[{label}] built in {meta['build_s']:.1f}s")
        table[label] = probe(label, runnable, model)
        del runnable
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
        time.sleep(2)

    # Static engines, one per length -- the cost baseline for dynamism.
    static = {}
    for seq in SEQ_LENS:
        try:
            r, m = build_static_onnx_trt(model, seq)
            x = bert_inputs(seq)
            t = common.bench_ms(lambda: r(x), warmup=10, rep=50)
            static[seq] = {"median_ms": t["median_ms"], "build_s": m["build_s"]}
            print(f"\n[static onnx_trt seq={seq}] {t['median_ms']:.3f}ms "
                  f"(built {m['build_s']:.1f}s)")
            del r
            gc.collect(); torch.cuda.empty_cache()
            time.sleep(2)
        except Exception as e:
            static[seq] = None
            print(f"\n[static onnx_trt seq={seq}] FAILED {type(e).__name__}")
    table["onnx_trt static (one engine each)"] = static

    for label, row in table.items():
        for seq, v in row.items():
            common.save_result("dynshape", {"route": label, "seq": seq,
                                            **(v or {"status": "failed"})})

    print("\n\n### latency by sequence length (bert_tiny, batch=8, fp16)\n")
    print("| route | " + " | ".join(f"seq={s}" for s in SEQ_LENS) + " |")
    print("|" + "---|" * (len(SEQ_LENS) + 1))
    for label, row in table.items():
        cells = [f"{row[s]['median_ms']:.3f}" if row.get(s) else "X"
                 for s in SEQ_LENS]
        print(f"| {label} | " + " | ".join(cells) + " |")

    print("\n### dynamic engine vs purpose-built static engine\n")
    print("| seq | dynamic onnx_trt | static onnx_trt | dynamic penalty |")
    print("|---|---|---|---|")
    dyn = table.get("onnx_trt dynamic", {})
    for s in SEQ_LENS:
        d, st = dyn.get(s), static.get(s)
        d_cell = f"{d['median_ms']:.3f}" if d else "X"
        s_cell = f"{st['median_ms']:.3f}" if st else "X"
        if d and st:
            pen = (d["median_ms"] / st["median_ms"] - 1) * 100
            print(f"| {s} | {d_cell} | {s_cell} | {pen:+.1f}% |")
        else:
            print(f"| {s} | {d_cell} | {s_cell} | - |")


if __name__ == "__main__":
    main()
