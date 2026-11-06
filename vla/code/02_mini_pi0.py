"""
02_mini_pi0.py —— 连续动作 + Flow Matching + Action Expert 路线（π0 / π0.5 / SmolVLA / GR00T）的最小实现

数据流（对照 openpi 源码 third_party/openpi/src/openpi/models/pi0.py）:

  ┌──────────── prefix（VLM 权重，d_vlm=192）────────────┐ ┌──── suffix（Action Expert 权重，d_exp=128）────┐
  image → TinyViT → proj ─┐                                 state ─ state_proj ─► [state token]
  instruction → embed ────┴► [img×64 | text×8]              noisy action chunk x_t (H,2) ─ action_in_proj ┐
                                                            τ ─ sincos ─────────────────────────────────┴ MLP ► [act×H]
                         embed_prefix()                                       embed_suffix()
  └───────────────────────────── 拼成一条序列，每层“各用各的权重，但在同一个 attention 里互相看” ───────────────┘
                                  (Mixture-of-Transformers, 见 MoTLayer)
  attention mask（block-causal，见 make_attn_mask）:
      prefix 内部双向；state 能看 prefix；action 能看 prefix+state+所有 action；prefix 看不到 suffix
  输出: suffix 最后 H 个 token → action_out_proj → 速度场 v_θ(x_t, τ | obs)
  训练: x_t = τ·ε + (1-τ)·a,  目标 u = ε - a,  loss = ‖v_θ - u‖²                             # pi0.py: compute_loss
  推理: 先跑一次 prefix 得到 KV cache；从 ε (τ=1) 出发做 10 步欧拉积分到 τ=0 得到动作块       # pi0.py: sample_actions
  执行: 一次预测 H=8 步动作，只执行前 4 步再重新规划（receding horizon / action chunking）

运行:  ../miniforge3/envs/vla/bin/python 02_mini_pi0.py
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import (
    QuantileNormalizer,
    TextTokenizer,
    TinyViT,
    ToyReachEnv,
    build_action_chunks,
    collect_demos,
    get_device,
    sinusoidal_embedding,
)


def make_attn_mask(pad_mask, ar_mask):
    """照搬 openpi 的 make_attn_mask：ar_mask 的累加和划分 block，token i 能看 token j 当且仅当 cumsum[j] <= cumsum[i]。
    pad_mask: (B, L) bool；ar_mask: (L,) bool。返回 (B, L, L) bool，True = 可见。"""
    cumsum = torch.cumsum(ar_mask.long(), dim=0)
    attn = cumsum[None, :] <= cumsum[:, None]  # (L, L)
    valid = pad_mask[:, None, :] & pad_mask[:, :, None]
    return attn[None] & valid


class MoTLayer(nn.Module):
    """一层 = 两套独立权重（VLM 专家 / Action 专家），但 Q/K/V 拼在一起做一次联合 attention。
    这就是 π0 里 PaliGemma(2B) + Gemma(300M) action expert 的结构（openpi models/gemma.py 中的多 config 模块）。"""

    def __init__(self, widths, n_heads=6, head_dim=32):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, head_dim
        inner = n_heads * head_dim
        self.norm1 = nn.ModuleList([nn.LayerNorm(w) for w in widths])
        self.qkv = nn.ModuleList([nn.Linear(w, 3 * inner) for w in widths])
        self.out = nn.ModuleList([nn.Linear(inner, w) for w in widths])
        self.norm2 = nn.ModuleList([nn.LayerNorm(w) for w in widths])
        self.mlp = nn.ModuleList([nn.Sequential(nn.Linear(w, 4 * w), nn.GELU(), nn.Linear(4 * w, w)) for w in widths])

    def forward(self, xs, mask, kv_cache=None):
        """xs: [prefix (B,Lp,d_vlm) 或 None, suffix (B,Ls,d_exp) 或 None]
        kv_cache: 推理时已缓存的 prefix (k, v)。返回 (新的 xs, 本层 prefix 的 (k,v))。"""
        qs, ks, vs, lens = [], [], [], []
        for i, x in enumerate(xs):
            if x is None:
                continue
            B, L, _ = x.shape
            q, k, v = self.qkv[i](self.norm1[i](x)).view(B, L, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            qs.append(q), ks.append(k), vs.append(v), lens.append((i, L))
        prefix_kv = (ks[0], vs[0]) if xs[0] is not None else None
        k, v, q = torch.cat(ks, 2), torch.cat(vs, 2), torch.cat(qs, 2)
        if kv_cache is not None:  # suffix 的 query 去看缓存的 prefix key/value
            k, v = torch.cat([kv_cache[0], k], 2), torch.cat([kv_cache[1], v], 2)
        h = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None])
        h = h.transpose(1, 2).flatten(2)

        outs, start = [None, None], 0
        for i, L in lens:  # 按 token 归属切回去，各自走自己的 out-proj 和 MLP
            x = xs[i] + self.out[i](h[:, start : start + L])
            outs[i] = x + self.mlp[i](self.norm2[i](x))
            start += L
        return outs, prefix_kv


class MiniPi0(nn.Module):
    def __init__(self, text_vocab, action_dim=2, state_dim=2, horizon=8, d_vlm=192, d_exp=128, d_vis=128, depth=4):
        super().__init__()
        self.horizon, self.action_dim = horizon, action_dim
        # --- prefix 侧（VLM）---
        self.vision = TinyViT(d=d_vis)
        self.img_proj = nn.Linear(d_vis, d_vlm)
        self.embed = nn.Embedding(text_vocab, d_vlm)
        # --- suffix 侧（Action Expert）---
        self.state_proj = nn.Linear(state_dim, d_exp)
        self.action_in_proj = nn.Linear(action_dim, d_exp)
        self.action_time_mlp_in = nn.Linear(2 * d_exp, d_exp)
        self.action_time_mlp_out = nn.Linear(d_exp, d_exp)
        self.action_out_proj = nn.Linear(d_exp, action_dim)
        # --- 共享 attention 的双专家 Transformer ---
        self.layers = nn.ModuleList([MoTLayer([d_vlm, d_exp]) for _ in range(depth)])
        self.pos_prefix = nn.Parameter(torch.randn(1, 64 + 8, d_vlm) * 0.02)
        self.pos_suffix = nn.Parameter(torch.randn(1, 1 + horizon, d_exp) * 0.02)

    def embed_prefix(self, image, text_ids, text_mask):
        img = self.img_proj(self.vision(image))  # (B, 64, d_vlm)
        txt = self.embed(text_ids)  # (B, 8, d_vlm)
        x = torch.cat([img, txt], 1)
        x = x + self.pos_prefix[:, : x.shape[1]]
        pad = torch.cat([torch.ones(img.shape[:2], dtype=torch.bool, device=x.device), text_mask], 1)
        ar = torch.zeros(x.shape[1], dtype=torch.bool, device=x.device)  # 全 0：prefix 内部双向注意力
        return x, pad, ar

    def embed_suffix(self, state, x_t, tau):
        B = state.shape[0]
        st = self.state_proj(state)[:, None]  # (B, 1, d_exp)
        act = self.action_in_proj(x_t)  # (B, H, d_exp)
        t_emb = sinusoidal_embedding(tau, act.shape[-1])[:, None].expand_as(act)
        act = self.action_time_mlp_out(F.silu(self.action_time_mlp_in(torch.cat([act, t_emb], -1))))
        x = torch.cat([st, act], 1) + self.pos_suffix
        pad = torch.ones(B, x.shape[1], dtype=torch.bool, device=x.device)
        # [True] 给 state：新 block 开始（prefix 看不到它）；action 的第一个为 True，其余 False：动作块内部双向
        ar = torch.tensor([True] + [True] + [False] * (self.horizon - 1), device=x.device)
        return x, pad, ar

    def forward_velocity(self, image, text_ids, text_mask, state, x_t, tau, verbose=False):
        """训练用：prefix + suffix 一次性前向。"""
        p, p_pad, p_ar = self.embed_prefix(image, text_ids, text_mask)
        s, s_pad, s_ar = self.embed_suffix(state, x_t, tau)
        mask = make_attn_mask(torch.cat([p_pad, s_pad], 1), torch.cat([p_ar, s_ar]))
        xs = [p, s]
        for layer in self.layers:
            xs, _ = layer(xs, mask)
        v = self.action_out_proj(xs[1][:, -self.horizon :])
        if verbose:
            print(f"    prefix tokens {tuple(p.shape)} (64 img + 8 text, d_vlm)")
            print(f"    suffix tokens {tuple(s.shape)} (1 state + {self.horizon} action, d_exp)")
            print(f"    attention mask {tuple(mask.shape)}；预测速度场 v {tuple(v.shape)}")
            print_mask_blocks(mask[0])
        return v

    def compute_loss(self, image, text_ids, text_mask, state, actions, is_pad):
        B = actions.shape[0]
        noise = torch.randn_like(actions)
        tau = torch.distributions.Beta(1.5, 1.0).sample((B,)).to(actions.device) * 0.999 + 0.001  # 偏向高噪声
        t = tau[:, None, None]
        x_t = t * noise + (1 - t) * actions  # 线性插值路径
        u_t = noise - actions  # 目标速度
        v_t = self.forward_velocity(image, text_ids, text_mask, state, x_t, tau)
        loss = ((v_t - u_t) ** 2).mean(-1)  # (B, H)
        return (loss * ~is_pad).sum() / (~is_pad).sum()  # 越过 episode 结尾的 padding 不计 loss

    @torch.no_grad()
    def sample_actions(self, image, text_ids, text_mask, state, num_steps=10):
        B, dev = state.shape[0], state.device
        # 1) prefix 只算一次，缓存每一层的 K/V（图像+语言很长，这是推理提速的关键）
        p, p_pad, p_ar = self.embed_prefix(image, text_ids, text_mask)
        p_mask = make_attn_mask(p_pad, p_ar)
        xs, cache = [p, None], []
        for layer in self.layers:
            xs, kv = layer(xs, p_mask)
            cache.append(kv)
        # 2) 从纯噪声开始，沿 τ: 1 -> 0 欧拉积分
        x_t, dt = torch.randn(B, self.horizon, self.action_dim, device=dev), -1.0 / num_steps
        for k in range(num_steps):
            tau = torch.full((B,), 1.0 + k * dt, device=dev)
            s, s_pad, s_ar = self.embed_suffix(state, x_t, tau)
            s_mask = make_attn_mask(s_pad, s_ar)  # suffix -> suffix
            full = torch.cat([p_pad[:, None, :].expand(-1, s.shape[1], -1), s_mask], -1)  # suffix -> [prefix|suffix]
            xs = [None, s]
            for layer, kv in zip(self.layers, cache):
                xs, _ = layer(xs, full, kv_cache=kv)
            v = self.action_out_proj(xs[1][:, -self.horizon :])
            x_t = x_t + dt * v
        return x_t  # τ=0 处即预测的（归一化）动作块


def print_mask_blocks(m):
    """把 (L, L) 的 mask 按 [img|text|state|action] 分块打印，看清楚谁能看谁。"""
    groups = {"img": range(0, 64), "text": range(64, 72), "state": range(72, 73), "action": range(73, m.shape[0])}
    print("    attention 可见性 (行=query, 列=key; ●=可见 ○=不可见 ◐=部分):")
    print("              " + "".join(f"{g:>8s}" for g in groups))
    for gq, rq in groups.items():
        row = []
        for gk, rk in groups.items():
            sub = m[list(rq)][:, list(rk)].float().mean().item()
            row.append("●" if sub > 0.99 else ("○" if sub < 0.01 else "◐"))
        print(f"      {gq:>8s}" + "".join(f"{c:>8s}" for c in row))


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--episodes", type=int, default=500)
    pa.add_argument("--steps", type=int, default=200)
    pa.add_argument("--bs", type=int, default=256)
    pa.add_argument("--horizon", type=int, default=8)
    pa.add_argument("--execute", type=int, default=4, help="每次规划后实际执行的步数")
    args = pa.parse_args()
    torch.manual_seed(0)
    device = get_device()
    os.makedirs("outputs", exist_ok=True)

    print("[1] 采集示教数据并构造 action chunk ...")
    data = collect_demos(args.episodes, seed=0)
    a_norm, s_norm = QuantileNormalizer(data["action"]), QuantileNormalizer(data["state"])
    chunks, is_pad = build_action_chunks(a_norm.normalize(data["action"]).astype(np.float32), data["episode_index"], args.horizon)
    tok = TextTokenizer()
    enc = [tok.encode(s) for s in data["instruction"]]
    T = lambda x, dt=torch.float32: torch.tensor(np.asarray(x), dtype=dt, device=device)
    images = T(data["image"], torch.uint8)
    text_ids, text_mask = T([e[0] for e in enc], torch.long), T([e[1] for e in enc], torch.bool)
    states = T(s_norm.normalize(data["state"]))
    chunks, is_pad = T(chunks), T(is_pad, torch.bool)
    print(f"    action chunks {tuple(chunks.shape)}  is_pad {tuple(is_pad.shape)}")

    model = MiniPi0(tok.vocab_size, horizon=args.horizon).to(device)
    print(f"[2] 模型参数量 {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M，一次前向:")
    model.forward_velocity(images[:4], text_ids[:4], text_mask[:4], states[:4], chunks[:4], torch.rand(4, device=device), verbose=True)

    print(f"[3] 训练 {args.steps} 步（flow matching loss）...")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=args.steps, pct_start=0.05)
    N = len(chunks)
    for step in range(args.steps):
        i = torch.randint(0, N, (args.bs,), device=device)
        loss = model.compute_loss(images[i], text_ids[i], text_mask[i], states[i], chunks[i], is_pad[i])
        opt.zero_grad(), loss.backward(), opt.step(), sched.step()
        if step % 50 == 0 or step == args.steps - 1:
            print(f"    step {step:5d}  loss {loss.item():.4f}")
    model.eval()

    def plan(obs):
        ids, m = tok.encode(obs["instruction"])
        chunk = model.sample_actions(
            T(obs["image"], torch.uint8)[None], T(ids, torch.long)[None], T(m, torch.bool)[None], T(s_norm.normalize(obs["state"]))[None]
        )[0].cpu().numpy()
        return a_norm.unnormalize(chunk)

    print("[4] 去噪过程示例：同一个观测，从噪声积分到动作")
    env = ToyReachEnv(7)
    obs = env.reset()
    print(f"    指令 {obs['instruction']!r}；专家第一步动作 {np.round(env.expert_action(), 4)}")
    print(f"    模型预测动作块前 3 步:\n{np.round(plan(obs)[:3], 4)}")

    print(f"[5] 闭环评测：每次预测 {args.horizon} 步，执行前 {args.execute} 步后重新规划")
    succ, n = 0, 5
    env = ToyReachEnv(1234)
    for _ in range(n):
        obs, done = env.reset(), False
        while not done:
            for a in plan(obs)[: args.execute]:
                obs, s, done = env.step(a)
                if done:
                    break
        succ += s
    print(f"    {n} 个 episode 闭环跑通，成功率 {succ / n:.0%}（训练步数很少，成功率低是正常的，这里只验证流程）")
    torch.save(model.state_dict(), "outputs/mini_pi0.pt")


if __name__ == "__main__":
    main()
