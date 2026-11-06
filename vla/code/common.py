"""
common.py —— 两个迷你 VLA 共用的部件：
  1. ToyReachEnv     一个 2D "机器人" 环境：渲染 RGB 图像 + 给出语言指令 + 接收连续动作
  2. TextTokenizer   词级别分词器（替代 LLM 的 BPE tokenizer）
  3. 数据集生成       专家策略采集示教 (demonstrations)，等价于真实机器人上的遥操作数据
  4. QuantileNormalizer  动作/状态归一化（OpenVLA 用 q01/q99 → [-1, 1]）
  5. Transformer 基础块 + 迷你 ViT（替代 SigLIP / DINOv2）

任务设计要点：目标块的位置只出现在“图像”里，要去哪个颜色只出现在“语言”里，
所以模型必须同时用到 Vision + Language 才能输出正确的 Action —— 这正是 VLA 的本质。
"""

import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------------------------------
# 1. 环境
# ----------------------------------------------------------------------------------------------------
IMG = 64  # 图像分辨率 64x64（真实 VLA 一般是 224x224）
COLORS = {
    "red": (230, 40, 40),
    "green": (40, 200, 40),
    "blue": (50, 80, 240),
    "yellow": (230, 210, 30),
}
TEMPLATES = ["move to the {} block", "go to the {} block", "reach the {} one", "please touch the {} cube"]
MAX_DELTA = 0.1  # 每步每个维度最多移动 0.1（世界坐标范围 [0, 1]）
SUCCESS_DIST = 0.05
MAX_STEPS = 30
EXPERT_NOISE = 0.01  # 示教噪声（模拟人类遥操作不完美）


class ToyReachEnv:
    """观测 = {image: (64,64,3) uint8, state: (2,) 末端位置, instruction: str}；动作 = (dx, dy) 连续值。"""

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        names = self.rng.choice(list(COLORS), size=3, replace=False)
        while True:  # 三个目标互相隔开
            pos = self.rng.uniform(0.1, 0.9, size=(3, 2))
            d = np.linalg.norm(pos[:, None] - pos[None], axis=-1) + np.eye(3)
            if d.min() > 0.25:
                break
        self.targets = dict(zip(names, pos))
        self.goal_name = str(self.rng.choice(names))
        self.agent = self.rng.uniform(0.05, 0.95, size=2)
        self.instruction = random.choice(TEMPLATES).format(self.goal_name)
        self.t = 0
        return self._obs()

    def _obs(self):
        return {"image": self.render(), "state": self.agent.astype(np.float32).copy(), "instruction": self.instruction}

    def render(self):
        img = np.full((IMG, IMG, 3), 30, dtype=np.uint8)
        for name, p in self.targets.items():
            cx, cy = (p * (IMG - 1)).astype(int)
            img[max(cy - 3, 0) : cy + 4, max(cx - 3, 0) : cx + 4] = COLORS[name]
        ax, ay = (self.agent * (IMG - 1)).astype(int)
        yy, xx = np.ogrid[:IMG, :IMG]
        img[(yy - ay) ** 2 + (xx - ax) ** 2 <= 4] = (255, 255, 255)  # 白色圆点 = 机械臂末端
        return img

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -MAX_DELTA, MAX_DELTA)
        self.agent = np.clip(self.agent + action, 0.0, 1.0)
        self.t += 1
        dist = np.linalg.norm(self.agent - self.targets[self.goal_name])
        success = dist < SUCCESS_DIST
        done = success or self.t >= MAX_STEPS
        return self._obs(), success, done

    def expert_action(self):
        """“遥操作员”：朝目标走，带一点噪声（真实示教数据也不是完美的）。"""
        delta = self.targets[self.goal_name] - self.agent
        n = np.linalg.norm(delta)
        if n > 0.08:
            delta = delta / n * 0.08
        return (delta + self.rng.normal(0, EXPERT_NOISE, size=2)).astype(np.float32)


# ----------------------------------------------------------------------------------------------------
# 2. 分词器
# ----------------------------------------------------------------------------------------------------
class TextTokenizer:
    PAD, BOS, SEP = 0, 1, 2  # SEP 相当于 OpenVLA prompt 里的 "Out:"，之后开始输出动作 token

    def __init__(self, max_len=8):
        words = sorted({w for t in TEMPLATES for w in t.split() if w != "{}"} | set(COLORS))
        self.itos = ["<pad>", "<bos>", "<sep>"] + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.max_len = max_len

    @property
    def vocab_size(self):
        return len(self.itos)

    def encode(self, text):
        """返回 (ids, mask)，右侧补 PAD 到 max_len。"""
        ids = [self.stoi[w] for w in text.split()][: self.max_len]
        mask = [1] * len(ids) + [0] * (self.max_len - len(ids))
        ids = ids + [self.PAD] * (self.max_len - len(ids))
        return np.array(ids, dtype=np.int64), np.array(mask, dtype=bool)


# ----------------------------------------------------------------------------------------------------
# 3. 示教数据：episode -> 帧 -> (obs, action, action_chunk)
# ----------------------------------------------------------------------------------------------------
def collect_demos(n_episodes, seed=0):
    """返回按“帧”展平的字典。真实世界里对应 LeRobotDataset / RLDS 里的一条条 step。"""
    env = ToyReachEnv(seed)
    random.seed(seed)
    frames = {k: [] for k in ["image", "state", "instruction", "action", "episode_index", "frame_index"]}
    for ep in range(n_episodes):
        obs = env.reset()
        for t in range(MAX_STEPS):
            a = env.expert_action()
            frames["image"].append(obs["image"])
            frames["state"].append(obs["state"])
            frames["instruction"].append(obs["instruction"])
            frames["action"].append(a)
            frames["episode_index"].append(ep)
            frames["frame_index"].append(t)
            obs, success, done = env.step(a)
            if done:
                break
    out = {k: np.stack(v) if k != "instruction" else v for k, v in frames.items()}
    return out


def build_action_chunks(actions, episode_index, horizon):
    """对每一帧 t 取 actions[t : t+horizon]（同一 episode 内），越界部分补 0 并标记 is_pad。
    对应 LeRobot 里的 delta_timestamps / action_is_pad，以及 π0 的 action_horizon。"""
    N, D = actions.shape
    chunks = np.zeros((N, horizon, D), dtype=np.float32)
    is_pad = np.ones((N, horizon), dtype=bool)
    for i in range(N):
        for h in range(horizon):
            j = i + h
            if j < N and episode_index[j] == episode_index[i]:
                chunks[i, h] = actions[j]
                is_pad[i, h] = False
    return chunks, is_pad


class QuantileNormalizer:
    """OpenVLA 风格：用 1% / 99% 分位数把每个维度线性映射到 [-1, 1]（对离群点更稳）。
    参考 third_party/openvla/prismatic/extern/hf/modeling_prismatic.py: predict_action 里的反归一化。"""

    def __init__(self, x):
        self.q01 = np.quantile(x, 0.01, axis=0).astype(np.float32)
        self.q99 = np.quantile(x, 0.99, axis=0).astype(np.float32)

    def normalize(self, x):
        return np.clip(2 * (x - self.q01) / (self.q99 - self.q01 + 1e-8) - 1, -1, 1)

    def unnormalize(self, x):
        return 0.5 * (x + 1) * (self.q99 - self.q01) + self.q01


# ----------------------------------------------------------------------------------------------------
# 4. Transformer 基础块
# ----------------------------------------------------------------------------------------------------
class Block(nn.Module):
    """Pre-norm Transformer block。attn_mask: (B, L, L) bool，True = 可以看见。"""

    def __init__(self, d, n_heads, mlp_ratio=4):
        super().__init__()
        self.n_heads = n_heads
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, mlp_ratio * d), nn.GELU(), nn.Linear(mlp_ratio * d, d))

    def forward(self, x, attn_mask=None):
        B, L, D = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(B, L, 3, self.n_heads, D // self.n_heads).permute(2, 0, 3, 1, 4)
        mask = None if attn_mask is None else attn_mask[:, None]
        h = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.proj(h.transpose(1, 2).reshape(B, L, D))
        return x + self.mlp(self.norm2(x))


class TinyViT(nn.Module):
    """视觉编码器（相当于 SigLIP / DINOv2）：64x64 图像 -> 8x8=64 个 patch token。"""

    def __init__(self, d=128, depth=2, n_heads=4, patch=8):
        super().__init__()
        self.patch = patch
        self.patch_embed = nn.Linear(3 * patch * patch, d)  # 等价于 Conv2d(3, d, kernel=patch, stride=patch)
        self.n_patches = (IMG // patch) ** 2
        self.pos = nn.Parameter(torch.randn(1, self.n_patches, d) * 0.02)
        self.blocks = nn.ModuleList([Block(d, n_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)

    def forward(self, img_uint8):  # (B, 64, 64, 3) uint8
        x = img_uint8.permute(0, 3, 1, 2).float() / 127.5 - 1.0  # -> (B, 3, 64, 64) in [-1, 1]
        x = F.unfold(x, kernel_size=self.patch, stride=self.patch).transpose(1, 2)  # -> (B, 64, 3*8*8) 切成 patch
        x = self.patch_embed(x) + self.pos  # -> (B, 64, d)
        for blk in self.blocks:
            x = blk(x)  # ViT 内部是双向注意力
        return self.norm(x)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sinusoidal_embedding(t, dim, min_period=4e-3, max_period=4.0):
    """π0 的时间步编码（posemb_sincos），t ∈ [0, 1]。"""
    frac = torch.linspace(0, 1, dim // 2, device=t.device)
    period = min_period * (max_period / min_period) ** frac
    ang = t[:, None] * (2 * math.pi / period[None])
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
