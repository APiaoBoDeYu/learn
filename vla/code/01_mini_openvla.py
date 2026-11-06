"""
01_mini_openvla.py —— 离散动作 token 路线（RT-2 / OpenVLA）的最小实现

数据流（对照 OpenVLA 源码）:
  image (B,64,64,3)
    └─ TinyViT ───────────────► patch tokens (B,64,d_vis)        # prismatic/extern/hf/modeling_prismatic.py: PrismaticVisionBackbone
        └─ Projector(MLP) ────► visual embeds (B,64,d)            # 同文件: PrismaticProjector (fc1-GELU-fc2)
  instruction "go to the red block"
    └─ tokenizer + embedding ─► text embeds (B,1+8+1,d)           # [BOS] words... [SEP]   (SEP ≈ "Out:")
  action (B,2) 连续值
    └─ 归一化到[-1,1] → 256 个 bin → 词表最后 256 个 id         # prismatic/vla/action_tokenizer.py
        └─ embedding ─────────► action embeds (B,2,d)
  拼接: [visual | text | action] → Causal Transformer (LLM) → LM head 预测下一个 token
  Loss: 只在 action token 位置算交叉熵（其余 label = -100）     # prismatic/vla/datasets/datasets.py: RLDSBatchTransform
  推理: 自回归生成 2 个 token → bin center → 反归一化 → (dx,dy)  # modeling_prismatic.py: predict_action

运行:  ../miniforge3/envs/vla/bin/python 01_mini_openvla.py
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import Block, QuantileNormalizer, TextTokenizer, TinyViT, ToyReachEnv, collect_demos, get_device

IGNORE_INDEX = -100


class ActionTokenizer:
    """连续动作 <-> 离散 token id。和 OpenVLA 一样：均匀分 bin，占用词表最后 n_bins 个 id。"""

    def __init__(self, text_vocab_size, n_bins=256):
        self.n_bins = n_bins
        self.bins = np.linspace(-1, 1, n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        self.vocab_size = text_vocab_size + n_bins  # 统一词表 = 文本词 + 动作 bin
        self.action_token_begin = text_vocab_size

    def encode(self, norm_action):  # (..., D) in [-1,1] -> (..., D) token id
        idx = np.digitize(np.clip(norm_action, -1, 1), self.bins)  # 1..n_bins
        return self.vocab_size - idx  # OpenVLA 的写法：vocab_size - bin_index

    def decode(self, token_ids):  # (..., D) token id -> (..., D) in [-1,1]
        idx = np.clip(self.vocab_size - token_ids - 1, 0, len(self.bin_centers) - 1)
        return self.bin_centers[idx]


class MiniOpenVLA(nn.Module):
    def __init__(self, vocab_size, action_dim=2, d=192, d_vis=128, depth=4, n_heads=6, max_len=128):
        super().__init__()
        self.action_dim = action_dim
        self.vision = TinyViT(d=d_vis)
        self.projector = nn.Sequential(nn.Linear(d_vis, d), nn.GELU(), nn.Linear(d, d))
        self.embed = nn.Embedding(vocab_size, d)  # LLM 的 token embedding（文本 + 动作共用）
        self.pos = nn.Parameter(torch.randn(1, max_len, d) * 0.02)
        self.blocks = nn.ModuleList([Block(d, n_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, vocab_size, bias=False)

    def forward(self, image, input_ids, attn_pad_mask, verbose=False):
        """input_ids: (B, T) = [BOS, text..., SEP, a_1, ..., a_D]；返回 logits (B, n_vis + T, V)"""
        vis = self.projector(self.vision(image))  # (B, 64, d)
        txt = self.embed(input_ids)  # (B, T, d)
        x = torch.cat([vis, txt], dim=1)
        B, L, _ = x.shape
        x = x + self.pos[:, :L]

        # 因果注意力 + padding mask（OpenVLA 的 Llama 就是纯 causal，图像 patch 放在最前面）
        causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=x.device))
        valid = torch.cat([torch.ones(B, vis.shape[1], dtype=torch.bool, device=x.device), attn_pad_mask], dim=1)
        mask = causal[None] & valid[:, None, :]
        mask |= torch.eye(L, dtype=torch.bool, device=x.device)[None]  # 防止 pad 行全 False 导致 NaN

        for blk in self.blocks:
            x = blk(x, mask)
        logits = self.lm_head(self.norm(x))
        if verbose:
            print(f"    image          {tuple(image.shape)}")
            print(f"    visual tokens  {tuple(vis.shape)}   (ViT patch -> projector)")
            print(f"    text+action ids{tuple(input_ids.shape)} -> embeds {tuple(txt.shape)}")
            print(f"    LLM input      {tuple(x.shape)}   attn mask {tuple(mask.shape)}")
            print(f"    logits         {tuple(logits.shape)}")
        return logits


def build_batch(tok, act_tok, instructions, norm_actions, device):
    """把 (指令, 动作) 变成 input_ids / labels。对应 OpenVLA 的 RLDSBatchTransform。"""
    ids, pads, labels = [], [], []
    for text, a in zip(instructions, norm_actions):
        t_ids, t_mask = tok.encode(text)
        a_ids = act_tok.encode(a)
        seq = np.concatenate([[tok.BOS], t_ids, [tok.SEP], a_ids])
        pad = np.concatenate([[True], t_mask, [True], np.ones(len(a_ids), bool)])
        lab = np.full_like(seq, IGNORE_INDEX)
        lab[-len(a_ids) :] = a_ids  # [CRITICAL] 只对动作 token 计算 loss
        ids.append(seq), pads.append(pad), labels.append(lab)
    f = lambda x, dt: torch.tensor(np.stack(x), dtype=dt, device=device)
    return f(ids, torch.long), f(pads, torch.bool), f(labels, torch.long)


def lm_loss(logits, labels, n_vis):
    """next-token 预测：位置 i 的 logits 预测位置 i+1 的 token。"""
    logits = logits[:, n_vis:]  # 只保留文本/动作部分
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=IGNORE_INDEX)


@torch.no_grad()
def predict_action(model, tok, act_tok, norm, obs, device):
    """自回归生成 action_dim 个 token（真实 OpenVLA 用 HF generate + KV cache）。"""
    t_ids, t_mask = tok.encode(obs["instruction"])
    ids = torch.tensor(np.concatenate([[tok.BOS], t_ids, [tok.SEP]]), device=device)[None]
    pad = torch.tensor(np.concatenate([[True], t_mask, [True]]), device=device)[None]
    img = torch.tensor(obs["image"], device=device)[None]
    for _ in range(model.action_dim):
        logits = model(img, ids, pad)[:, -1]
        logits[:, : act_tok.action_token_begin] = -1e9  # 约束解码：只允许动作 token
        nxt = logits.argmax(-1, keepdim=True)
        ids = torch.cat([ids, nxt], 1)
        pad = torch.cat([pad, torch.ones_like(nxt, dtype=torch.bool)], 1)
    a_ids = ids[0, -model.action_dim :].cpu().numpy()
    return norm.unnormalize(act_tok.decode(a_ids)), a_ids


def evaluate(policy_fn, n=5, seed=1234):
    env, succ = ToyReachEnv(seed), 0
    for _ in range(n):
        obs, done = env.reset(), False
        while not done:
            obs, s, done = env.step(policy_fn(obs))
        succ += s
    return succ / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--bins", type=int, default=256, help="每个动作维度的离散 bin 数（OpenVLA 用 256）")
    p.add_argument("--lr", type=float, default=3e-4)
    args = p.parse_args()
    torch.manual_seed(0)
    device = get_device()
    os.makedirs("outputs", exist_ok=True)

    print("[1] 采集示教数据 ...")
    data = collect_demos(args.episodes, seed=0)
    norm = QuantileNormalizer(data["action"])
    norm_actions = norm.normalize(data["action"])
    tok = TextTokenizer()
    act_tok = ActionTokenizer(tok.vocab_size, n_bins=args.bins)
    print(f"    {len(data['action'])} 帧；文本词表 {tok.vocab_size} + 动作 bin {act_tok.n_bins} = 统一词表 {act_tok.vocab_size}")

    a = data["action"][0]
    print(f"[2] 动作离散化示例: 原始 {np.round(a, 4)} -> 归一化 {np.round(norm_actions[0], 3)} "
          f"-> token {act_tok.encode(norm_actions[0])} -> 还原 {np.round(norm.unnormalize(act_tok.decode(act_tok.encode(norm_actions[0]))), 4)}")

    model = MiniOpenVLA(act_tok.vocab_size).to(device)
    print(f"[3] 模型参数量 {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M，一次前向的张量形状:")
    images = torch.tensor(data["image"], device=device)
    ids, pad, labels = build_batch(tok, act_tok, data["instruction"][:4], norm_actions[:4], device)
    model(images[:4], ids, pad, verbose=True)
    print(f"    labels[0] = {labels[0].tolist()}   (-100 = 不算 loss)")

    print(f"[4] 训练 {args.steps} 步 ...")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    N = len(norm_actions)
    for step in range(args.steps):
        idx = np.random.randint(0, N, args.bs)
        ids, pad, labels = build_batch(tok, act_tok, [data["instruction"][i] for i in idx], norm_actions[idx], device)
        logits = model(images[idx], ids, pad)
        loss = lm_loss(logits, labels, model.vision.n_patches)
        opt.zero_grad(), loss.backward(), opt.step(), sched.step()
        if step % 50 == 0 or step == args.steps - 1:
            print(f"    step {step:5d}  loss {loss.item():.4f}")

    model.eval()
    print("[5] 闭环评测（模型输出动作 -> 环境执行 -> 新观测 -> ...）")
    env = ToyReachEnv(7)
    obs = env.reset()
    act, a_ids = predict_action(model, tok, act_tok, norm, obs, device)
    print(f"    指令 {obs['instruction']!r}  生成 token {a_ids.tolist()} -> 动作 {np.round(act, 4)}  专家动作 {np.round(env.expert_action(), 4)}")
    sr = evaluate(lambda o: predict_action(model, tok, act_tok, norm, o, device)[0])
    print(f"    5 个 episode 闭环跑通，成功率 {sr:.0%}（训练步数很少，成功率低是正常的，这里只验证流程）")
    torch.save(model.state_dict(), "outputs/mini_openvla.pt")


if __name__ == "__main__":
    main()
