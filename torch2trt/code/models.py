"""The model zoo used by every route in this tutorial.

Why these four models and not one: the whole point of the tutorial is that the
*ranking* of the conversion routes changes with the model, so a single-model
benchmark would give a misleading answer. Each model here is chosen to stress a
different property of the conversion pipeline.

  resnet18   - static shapes, pure conv/bn/relu. Every route supports it and
               TensorRT's conv+bn+relu fusion is at its best. This is the
               "everything works" control case, and the ceiling for TRT's win.

  bert_tiny  - transformer encoder block stack. matmul + layernorm + softmax
               heavy, and (with dynamic seq len) the case where shape handling
               separates the routes: ONNX needs explicit dynamic axes, the
               dynamo route needs torch.export Dim, trtexec needs opt profiles.

  ctrlflow   - a small net with a data-dependent `if` on a tensor value. This
               is the interesting one: it CANNOT be traced, so it exposes the
               real difference between trace-based export (ONNX/TorchScript
               trace silently bake one branch) and graph-break-tolerant routes
               (torch.compile just breaks the graph and stays correct).

  customop   - a net containing an op with no ONNX/TensorRT converter. Shows
               what each route does when it hits an unsupported op: hard
               failure, plugin requirement, or automatic partial fallback.

Every builder returns (model, example_inputs, kwargs_for_export) so the route
modules can stay generic.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# 1. resnet18 -- the control case
# --------------------------------------------------------------------------
def resnet18(batch: int = 8, dtype=torch.half):
    import torchvision
    # weights=None keeps this offline-runnable; we only measure latency and
    # route-vs-route numerical agreement, never real accuracy, so random
    # weights are fine. (If you do want real top-1, pass weights="DEFAULT".)
    model = torchvision.models.resnet18(weights=None).eval().to("cuda", dtype)
    # generate in fp32 then cast, so the RNG draw is identical whatever
    # dtype we ask for -- see get()'s docstring
    x = torch.randn(batch, 3, 224, 224, device="cuda").to(dtype)
    return model, (x,)


# --------------------------------------------------------------------------
# 2. bert_tiny -- transformer, the dynamic-shape case
# --------------------------------------------------------------------------
class EncoderBlock(nn.Module):
    """One pre-norm transformer encoder block, written out by hand.

    Deliberately not `nn.TransformerEncoderLayer`: that module contains
    branching on `is_causal`/mask types and calls the fused SDPA path, which
    makes the exported graph differ between routes and muddies the comparison.
    A hand-written block exports identically everywhere.
    """

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.h = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.fc1, self.fc2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def forward(self, x):
        B, S, D = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        # (B, S, D) -> (B, heads, S, head_dim)
        q, k, v = (t.view(B, S, self.h, D // self.h).transpose(1, 2) for t in (q, k, v))
        att = F.scaled_dot_product_attention(q, k, v)
        att = att.transpose(1, 2).reshape(B, S, D)
        x = x + self.proj(att)
        h = self.ln2(x)
        return x + self.fc2(F.gelu(self.fc1(h)))


class BertTiny(nn.Module):
    def __init__(self, vocab=30522, d=128, layers=4, heads=2, max_len=512):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([EncoderBlock(d, heads) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, 2)

    def forward(self, ids):
        S = ids.shape[1]
        x = self.tok(ids) + self.pos(torch.arange(S, device=ids.device))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x)[:, 0])


def bert_tiny(batch: int = 8, seq: int = 128, dtype=torch.half):
    model = BertTiny().eval().to("cuda", dtype)
    # int64 token ids -- note this is the input that trips `truncate_double` /
    # int64->int32 handling in the TensorRT routes; see chapter 03.
    ids = torch.randint(0, 30000, (batch, seq), device="cuda", dtype=torch.long)
    return model, (ids,)


# --------------------------------------------------------------------------
# 3. ctrlflow -- data-dependent control flow
# --------------------------------------------------------------------------
class DataDependentBranch(nn.Module):
    """Branches on a *value computed from the input*, not on a Python config.

    `x.mean() > 0` cannot be resolved at export time, so:
      - torch.jit.trace / ONNX trace: records only the branch taken by the
        example input, silently producing a wrong model for other inputs.
      - torch.export: raises (or specializes with a guard) rather than lying.
      - torch.compile: graph-breaks at the `if` and stays correct.
    Chapter 05 turns this into a measured experiment, not just an assertion.
    """

    def __init__(self, d: int = 512):
        super().__init__()
        self.a = nn.Linear(d, d)
        self.b = nn.Linear(d, d)
        self.out = nn.Linear(d, 10)

    def forward(self, x):
        if x.mean() > 0:
            x = F.relu(self.a(x))
        else:
            x = torch.tanh(self.b(x))
        return self.out(x)


def ctrlflow(batch: int = 32, dtype=torch.half):
    model = DataDependentBranch().eval().to("cuda", dtype)
    x = torch.randn(batch, 512, device="cuda").to(dtype)
    return model, (x,)


# --------------------------------------------------------------------------
# 4. customop -- an op with no ONNX/TRT converter
# --------------------------------------------------------------------------
class NoisyActivation(nn.Module):
    """Contains `torch.erfinv`, which has no ONNX opset mapping.

    erfinv is a real aten op (so eager and torch.compile handle it fine) but it
    is not in the ONNX operator set and has no Torch-TensorRT converter. That
    makes it a clean, non-contrived probe of each route's unsupported-op
    behaviour, which chapter 05 measures.
    """

    def __init__(self, d: int = 512):
        super().__init__()
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, 10)

    def forward(self, x):
        x = self.fc1(x)
        # clamp keeps erfinv in its finite domain (-1, 1)
        x = torch.erfinv(x.clamp(-0.99, 0.99))
        return self.fc2(x)


def customop(batch: int = 32, dtype=torch.half):
    model = NoisyActivation().eval().to("cuda", dtype)
    x = torch.randn(batch, 512, device="cuda").to(dtype)
    return model, (x,)


ZOO = {
    "resnet18": resnet18,
    "bert_tiny": bert_tiny,
    "ctrlflow": ctrlflow,
    "customop": customop,
}


def get(name: str, seed: int = 0, **kw):
    """Build a model by name. Returns (model, example_inputs).

    The seed is reset before *both* the weight init and the input generation, so
    repeated calls return bit-identical weights and inputs. This is not a nicety
    — the accuracy check compares a route's output against a separately-built
    fp32 reference model, and without the seed those two models would have
    different random weights and every cosine similarity would read ~0.
    """
    if name not in ZOO:
        raise KeyError(f"unknown model {name!r}; choose from {sorted(ZOO)}")
    torch.manual_seed(seed)
    return ZOO[name](**kw)


if __name__ == "__main__":
    import common
    common.device_info()
    for name in ZOO:
        model, inputs = get(name)
        with torch.no_grad():
            out = model(*inputs)
        n_params = sum(p.numel() for p in model.parameters())
        shapes = [tuple(i.shape) for i in inputs]
        print(f"{name:10s} in={shapes} out={tuple(out.shape)} "
              f"params={n_params/1e6:.2f}M dtype={out.dtype}")
