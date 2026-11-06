"""
00_explore_data.py —— 先看数据：一条 VLA 训练样本长什么样？

运行:  ../miniforge3/envs/vla/bin/python 00_explore_data.py
产出:  outputs/sample_frames.png
"""

import os

import numpy as np

from common import QuantileNormalizer, TextTokenizer, build_action_chunks, collect_demos

os.makedirs("outputs", exist_ok=True)
data = collect_demos(n_episodes=20, seed=0)

print("=== 1. 原始示教数据（按帧展平，类似 LeRobotDataset 的一行一帧）===")
for k, v in data.items():
    if k == "instruction":
        print(f"  {k:14s} list[str] len={len(v)}  例: {v[0]!r}")
    else:
        print(f"  {k:14s} {str(v.dtype):8s} shape={v.shape}")

ep0 = data["episode_index"] == 0
print(f"\n第 0 条 episode 有 {ep0.sum()} 帧，指令: {data['instruction'][0]!r}")
print("  前 3 帧 state :", np.round(data["state"][ep0][:3].astype(float), 3).tolist())
print("  前 3 帧 action:", np.round(data["action"][ep0][:3].astype(float), 3).tolist())

print("\n=== 2. 归一化（OpenVLA: q01/q99 -> [-1,1]）===")
norm = QuantileNormalizer(data["action"])
print("  q01 =", norm.q01, " q99 =", norm.q99)
a = data["action"][0]
print(f"  原始 {a} -> 归一化 {norm.normalize(a)} -> 反归一化 {norm.unnormalize(norm.normalize(a))}")

print("\n=== 3. 语言分词 ===")
tok = TextTokenizer()
ids, mask = tok.encode(data["instruction"][0])
print(f"  {data['instruction'][0]!r}\n  ids ={ids.tolist()}\n  mask={mask.astype(int).tolist()}")

print("\n=== 4. 动作分块 action chunk（π0 / ACT / SmolVLA 用）===")
chunks, is_pad = build_action_chunks(data["action"], data["episode_index"], horizon=8)
last = np.where(ep0)[0][-1]
print(f"  chunks shape={chunks.shape}, is_pad shape={is_pad.shape}")
print(f"  episode 0 最后一帧的 is_pad: {is_pad[last].astype(int).tolist()}  (越过 episode 结尾的部分被 mask 掉)")

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = np.where(ep0)[0]
    pick = idx[np.linspace(0, len(idx) - 1, 5).astype(int)]
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.4))
    for ax, i in zip(axes, pick):
        ax.imshow(data["image"][i])
        dx, dy = data["action"][i] * 63 * 3
        x, y = data["state"][i] * 63
        ax.arrow(x, y, dx, dy, color="white", head_width=2)
        ax.set_title(f"t={data['frame_index'][i]}  a=({data['action'][i][0]:+.2f},{data['action'][i][1]:+.2f})", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"instruction: {data['instruction'][0]!r}   (white dot = end effector, arrow = expert action)")
    plt.tight_layout()
    plt.savefig("outputs/sample_frames.png", dpi=100)
    print("\n已保存 outputs/sample_frames.png")
except ImportError:
    print("\n(未安装 matplotlib，跳过画图)")
