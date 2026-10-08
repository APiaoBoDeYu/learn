"""Experiment: how each route handles data-dependent control flow.

This is the experiment that separates the routes on *correctness* rather than
speed, and it is the single most important one in the tutorial — a route that is
2x faster but silently wrong is not a faster route.

`DataDependentBranch.forward` does `if x.mean() > 0`. There is no way to bake
that into a static graph, so every route has to do *something*. The three
possible somethings, and what they mean for you:

  1. Silently specialize (trace-based: torch.jit.trace, legacy ONNX export).
     The export succeeds. The artifact is wrong for half of all inputs and
     nothing warns you. This is the dangerous outcome.
  2. Refuse (torch.export / AOTInductor). Build fails loudly. Annoying, honest.
  3. Graph-break and stay correct (torch.compile). Both branches keep working
     because the `if` runs in Python; you lose the fusion across that boundary.

The script proves which bucket each route is in by evaluating with a positive-mean
input AND a negative-mean input and checking both against eager.
"""
import torch

import common
import models
import routes


def probe(route_name: str, model, pos_in, neg_in, ref_pos, ref_neg) -> dict:
    """Build via one route, then test BOTH branches."""
    print(f"\n--- {route_name} ---")
    try:
        runnable, meta = routes.ROUTES[route_name](model, (pos_in,), name="ctrlflow")
    except routes.RouteUnsupported as e:
        print(f"  build REFUSED: {str(e)[:160]}")
        return {"route": route_name, "outcome": "refused", "error": str(e)[:300]}

    with torch.no_grad():
        try:
            out_pos = runnable(pos_in)
            out_neg = runnable(neg_in)
        except Exception as e:
            print(f"  runtime error: {type(e).__name__}: {e}")
            return {"route": route_name, "outcome": "runtime_error",
                    "error": str(e)[:300]}

    def agree(a, b):
        a = (a[0] if isinstance(a, (tuple, list)) else a).float().cpu()
        return torch.nn.functional.cosine_similarity(
            a.flatten(), b.flatten(), dim=0).item()

    cos_pos, cos_neg = agree(out_pos, ref_pos), agree(out_neg, ref_neg)
    # The traced-branch signature: the branch that WAS traced matches, the other
    # one does not. A correct route matches both.
    if cos_pos > 0.99 and cos_neg > 0.99:
        outcome = "correct_both_branches"
    elif cos_pos > 0.99:
        outcome = "SILENTLY_WRONG_on_untraced_branch"
    else:
        outcome = "wrong_on_both"
    print(f"  cos(pos-mean input)={cos_pos:.5f}  cos(neg-mean input)={cos_neg:.5f}")
    print(f"  => {outcome}")
    return {"route": route_name, "outcome": outcome,
            "cos_traced_branch": cos_pos, "cos_other_branch": cos_neg,
            "build_s": meta.get("build_s")}


def main():
    common.device_info()
    model, (x,) = models.get("ctrlflow")

    # Two inputs that take opposite branches. We shift the mean rather than
    # reroll, so shape/scale stay comparable and only the branch differs.
    pos_in = x.abs()          # mean > 0  -> the `relu(self.a(x))` branch
    neg_in = -x.abs()         # mean < 0  -> the `tanh(self.b(x))` branch
    with torch.no_grad():
        assert pos_in.mean().item() > 0 and neg_in.mean().item() < 0
        ref_pos = model(pos_in).float().cpu()
        ref_neg = model(neg_in).float().cpu()
    # Sanity: the two branches must actually produce different outputs, else
    # this experiment proves nothing.
    branch_gap = torch.nn.functional.cosine_similarity(
        ref_pos.flatten(), ref_neg.flatten(), dim=0).item()
    print(f"cos(eager pos-branch, eager neg-branch) = {branch_gap:.5f}"
          "   (must be well below 1, or the test is vacuous)")

    rows = []
    for r in ["eager", "torch_compile", "ts_trt", "dynamo_trt", "onnx_trt", "aoti"]:
        row = probe(r, model, pos_in, neg_in, ref_pos, ref_neg)
        rows.append(row)
        common.save_result("ctrlflow_experiment", row)

    print("\n\n### control-flow verdict\n")
    print("| route | outcome | cos(traced branch) | cos(other branch) |")
    print("|---|---|---|---|")
    for r in rows:
        cp = f"{r['cos_traced_branch']:.5f}" if "cos_traced_branch" in r else "-"
        cn = f"{r['cos_other_branch']:.5f}" if "cos_other_branch" in r else "-"
        print(f"| {r['route']} | {r['outcome']} | {cp} | {cn} |")


if __name__ == "__main__":
    main()
