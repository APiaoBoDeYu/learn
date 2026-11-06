"""07-05 VLA / LLM 推理专项：prefill vs decode、KV cache、batch、CUDA Graph

    /opt/venv/bin/python 07_optimization/05_llm_decode.py

模型：GPT 式 decoder，d=1024、16 头、8 层、MLP 4d，fp16，约 1.0 亿参数（权重约 201 MB）。
    （ToyVLA 的动作头只输出 16 个动作，不做自回归；真实的 VLA 用语言模型做 backbone，decode 的特性与此相同）
- KV cache 预分配到 max_len，写入位置和 attention mask 都是 GPU 张量 → decode 的一步可以被 CUDA Graph 捕获
A. prefill（一次处理 512 个 token）vs decode（每次 1 个 token）：时间、算力、权重读取带宽
B. 生成 128 个 token：每步重新计算整个序列（无 KV cache）vs KV cache
C. decode 的 batch：1 / 4 / 16 / 64 → 每步时间与 token/s
D. decode 一步：eager vs CUDA Graph
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import benchmark  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs"
D, H, NL, MAXLEN = 1024, 16, 8, 1024
HD = D // H
PEAK_BW, PEAK_FP16 = 240.0, 155.6     # 第 00 章实测（GB/s、TFLOPS）
lines = []


def p(s=""):
    print(s); lines.append(s)


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.qkv, self.o = nn.Linear(D, 3 * D), nn.Linear(D, D)
        self.fc1, self.fc2 = nn.Linear(D, 4 * D), nn.Linear(4 * D, D)

    def forward(self, x, kc, vc, pos, mask):
        """x (B,T,D)；kc/vc (B,H,MAXLEN,HD)；pos (T,) 写入位置；mask (B,1,T,MAXLEN) bool，True = 可以看"""
        B, T, _ = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, H, HD).permute(2, 0, 3, 1, 4)
        kc.index_copy_(2, pos, k)
        vc.index_copy_(2, pos, v)
        a = F.scaled_dot_product_attention(q, kc, vc, attn_mask=mask)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, D))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(Layer() for _ in range(NL))
        self.head = nn.Linear(D, D)                  # 代替词表投影（与本节结论无关，保持模型小）

    def forward(self, x, caches, pos, mask):
        for l, (kc, vc) in zip(self.layers, caches):
            x = l(x, kc, vc, pos, mask)
        return self.head(x)


def new_cache(B):
    return [(torch.zeros(B, H, MAXLEN, HD, device="cuda", dtype=torch.half),
             torch.zeros(B, H, MAXLEN, HD, device="cuda", dtype=torch.half)) for _ in range(NL)]


def causal_mask(B, start, T):
    """第 start..start+T-1 个 token 可以看到位置 0..自己。"""
    q = torch.arange(start, start + T, device="cuda")[:, None]
    kpos = torch.arange(MAXLEN, device="cuda")[None, :]
    return (kpos <= q).expand(B, 1, T, MAXLEN).contiguous()


def main():
    torch.manual_seed(0)
    m = TinyLM().cuda().half().eval()
    n_params = sum(x.numel() for x in m.parameters())
    wbytes = n_params * 2
    p(f"模型参数 {n_params / 1e6:.1f} M，fp16 权重 {wbytes / 1e6:.0f} MB；decode 一步至少要把权重读一遍："
      f"{wbytes / 1e6:.0f} MB / {PEAK_BW:.0f} GB/s = {wbytes / PEAK_BW / 1e6:.3f} ms（带宽下限）")
    with torch.no_grad():
        # ---------------- A
        p("\n== A. prefill vs decode（batch=1）==")
        P = 512
        caches = new_cache(1)
        x = torch.randn(1, P, D, device="cuda", dtype=torch.half)
        pos, mask = torch.arange(P, device="cuda"), causal_mask(1, 0, P)
        t_pre = benchmark(lambda: m(x, caches, pos, mask), mode="event", iters=30).p50
        x1 = torch.randn(1, 1, D, device="cuda", dtype=torch.half)
        pos1, mask1 = torch.tensor([P], device="cuda"), causal_mask(1, P, 1)
        t_dec = benchmark(lambda: m(x1, caches, pos1, mask1), mode="event", iters=50).p50
        flop_tok = 2 * n_params                     # 每个 token 约 2*N FLOP（忽略 attention 的 L 相关项）
        for name, t, T in (("prefill 512 token", t_pre, P), ("decode 1 token", t_dec, 1)):
            tf = flop_tok * T / t / 1e9
            p(f"  {name:<18s} {t:8.3f} ms  {T / t * 1000:9.0f} token/s  {tf:6.2f} TFLOPS（峰值的 {100 * tf / PEAK_FP16:4.1f}%）"
              f"  权重读取带宽 {wbytes / t / 1e6:6.1f} GB/s（峰值的 {100 * wbytes / t / 1e6 / PEAK_BW:4.0f}%）")
        p(f"  → prefill 每个 token 的 AI ≈ {flop_tok * P / wbytes:.0f} FLOP/B（Compute-bound 一侧），"
          f"decode ≈ {flop_tok / wbytes:.0f} FLOP/B（远低于拐点约 650，Memory-bound / Launch-bound）")

        # ---------------- B
        p("\n== B. 生成 128 个 token（batch=1）：无 KV cache（每步重算整个序列）vs KV cache ==")
        gen = 128
        for prompt in (128, 512):
            def no_cache():
                cs = new_cache(1)
                seq = torch.randn(1, prompt, D, device="cuda", dtype=torch.half)
                for i in range(gen):
                    L = prompt + i
                    out = m(seq, cs, torch.arange(L, device="cuda"), causal_mask(1, 0, L))
                    seq = torch.cat([seq, out[:, -1:]], dim=1)          # 每一步都重新计算整个序列
                return seq

            def with_cache():
                cs = new_cache(1)
                seq = torch.randn(1, prompt, D, device="cuda", dtype=torch.half)
                out = m(seq, cs, torch.arange(prompt, device="cuda"), causal_mask(1, 0, prompt))
                tok = out[:, -1:]
                for i in range(gen):
                    L = prompt + i
                    tok = m(tok, cs, torch.tensor([L], device="cuda"), causal_mask(1, L, 1))
                return tok
            t_nc = benchmark(no_cache, iters=3, warmup=1).p50
            t_c = benchmark(with_cache, iters=3, warmup=1).p50
            p(f"  prompt {prompt:3d}：无 KV cache {t_nc:7.1f} ms（{gen / t_nc * 1000:4.0f} token/s）  "
              f"KV cache {t_c:7.1f} ms（{gen / t_c * 1000:4.0f} token/s）  {t_nc / t_c:.2f}x")

        # ---------------- C
        p("\n== C. decode 的 batch（KV 长度 512，eager）==")
        p(f"  {'batch':>5s} {'每步 ms':>8s} {'token/s':>9s} {'相对 batch=1 的吞吐':>18s}")
        base = None
        for B in (1, 4, 16, 64):
            cs = new_cache(B)
            xb = torch.randn(B, 1, D, device="cuda", dtype=torch.half)
            mb = causal_mask(B, P, 1)
            t = benchmark(lambda: m(xb, cs, pos1, mb), mode="event", iters=50).p50
            thr = B / t * 1000
            base = base or thr
            p(f"  {B:5d} {t:8.3f} {thr:9.0f} {thr / base:17.1f}x")
            del cs

        # ---------------- D
        p("\n== D. decode 一步：eager vs CUDA Graph（batch=1）==")
        cs = new_cache(1)
        xg = torch.randn(1, 1, D, device="cuda", dtype=torch.half)
        posg, maskg = torch.tensor([P], device="cuda"), causal_mask(1, P, 1)
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                m(xg, cs, posg, maskg)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            yg = m(xg, cs, posg, maskg)
        ref = m(xg, cs, posg, maskg).clone()
        g.replay(); torch.cuda.synchronize()
        assert torch.allclose(yg, ref, atol=1e-2)
        t_e = benchmark(lambda: m(xg, cs, posg, maskg), mode="event", iters=100).p50
        t_w = benchmark(lambda: m(xg, cs, posg, maskg), mode="wall", iters=100).p50
        t_g = benchmark(g.replay, mode="event", iters=100).p50
        p(f"  eager        GPU {t_e:.3f} ms（wall {t_w:.3f} ms）")
        p(f"  CUDA Graph   GPU {t_g:.3f} ms   {t_e / t_g:.2f}x；权重读取带宽 {wbytes / t_g / 1e6:.0f} GB/s"
          f"（带宽下限 {wbytes / PEAK_BW / 1e6:.3f} ms）")
    (OUT / "05_llm_decode.log").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
