# VLA（Vision-Language-Action）快速入门教程

> 面向：有 Transformer / 深度学习基础，刚接触机器人和 VLA 的同学
> 目标：搞清楚 **VLA 的原理、两大主流架构、数据从相机到电机经过了哪些步骤**，并能读懂 OpenVLA / π0 / SmolVLA 的源码
> 方式：先读概念 → 跑两个“迷你 VLA”（在你机器上几分钟就能训完） → 再对照真实仓库源码

```
/root/data/learn/vla/
├── TUTORIAL.md                ← 本文
├── code/
│   ├── common.py              玩具环境 / 分词器 / 归一化 / ViT 等公共部件
│   ├── 00_explore_data.py     第 1 课：看清一条训练样本
│   ├── 01_mini_openvla.py     第 2 课：离散 token 路线（RT-2 / OpenVLA）
│   ├── 02_mini_pi0.py         第 3 课：Flow Matching + Action Expert 路线（π0 / SmolVLA）
│   └── outputs/               图片和模型权重
├── third_party/               真实仓库（只读参考）
│   ├── openvla/   lerobot/   openpi/
└── miniforge3/envs/vla        独立的 conda 环境（没有动 /opt/venv）
```

运行方式：

```bash
cd /root/data/learn/vla/code
PY=/root/data/learn/vla/miniforge3/envs/vla/bin/python
$PY 00_explore_data.py
$PY 01_mini_openvla.py   # 约 25 秒：数据 → 前向 → loss → 训练 200 步 → 自回归解码 → 闭环执行
$PY 02_mini_pi0.py       # 约 25 秒：数据 → chunk → 前向 → flow matching loss → KV cache 去噪 → 闭环执行
```

默认参数只是为了**打通流程**，200 步训练得到的模型成功率很低，这是正常的。重点看每一步打印出来的张量形状和中间结果。

---

## 0. 一句话理解 VLA

**VLA = 一个预训练好的 VLM（视觉语言模型）+ 一个把输出变成机器人动作的“动作头”，用机器人示教数据做模仿学习（行为克隆）。**

形式化地说，VLA 学的是一个策略：

$$\pi_\theta(\;a_{t:t+H}\;\mid\; I_t,\; \ell,\; s_t\;)$$

| 符号 | 含义 | 真实例子 |
|---|---|---|
| $I_t$ | 当前的相机图像（可能有好几路） | 第三人称相机 + 腕部相机，224×224 RGB |
| $\ell$ | 语言指令 | "put the red cup in the sink" |
| $s_t$ | 本体状态 (proprioception) | 7 个关节角 + 夹爪开合 |
| $a_{t:t+H}$ | 未来 H 步的动作 | 末端位姿增量 (Δx,Δy,Δz,Δroll,Δpitch,Δyaw,gripper) |

和你熟悉的 LLM 对比：

| | LLM | VLM（LLaVA 等） | VLA |
|---|---|---|---|
| 输入 | 文本 | 图像 + 文本 | 图像 + 文本 + **机器人状态** |
| 输出 | 文本 token | 文本 token | **连续的机器人动作** |
| 训练数据 | 互联网文本 | 图文对 | **机器人示教轨迹**（遥操作采集） |
| 运行方式 | 一次生成 | 一次生成 | **闭环**：观测→动作→执行→新观测→…，频率 5~50Hz |

VLA 的核心难点都来自最后一列：

1. **动作是连续的高维向量**，而 LLM 只会输出离散 token → 怎么表示动作？（第 2、3 节的两大路线）
2. **数据少且贵**：机器人数据是百万级帧，互联网图文是十亿级 → 要继承 VLM 的预训练知识
3. **要实时**：大模型推理慢，机器人需要高频控制 → action chunking、KV cache、小的动作专家网络
4. **多模态分布**：同一个场景可以“从左边绕”也可以“从右边绕”，MSE 回归会取平均变成“撞上去” → 需要生成式动作头（diffusion / flow matching）或离散分类

---

## 1. 数据：一条 VLA 训练样本长什么样（第 1 课）

运行 `00_explore_data.py`，玩具任务如下（见 `outputs/sample_frames.png`）：

![sample](code/outputs/sample_frames.png)

- 画面里有 3 个彩色方块和一个白点（机械臂末端）
- 指令是 "go to the blue block" 这类句子
- 动作是 2 维连续向量 (dx, dy)
- **方块的位置只出现在图像里，要去哪个颜色只出现在语言里**：模型必须同时用上视觉和语言，这就是 VLA 最小的完整形态

### 1.1 数据层级：Dataset → Episode → Frame

```
数据集（例如 Open X-Embodiment，约 100 万条轨迹 / 22 种机器人）
 └─ episode（一次完整的示教，比如 “把杯子放进水槽”，几十到几百帧）
     └─ frame t：{ image(s), state, language_instruction, action, timestamp }
```

真实世界中有两种主流格式：

| 格式 | 谁在用 | 特点 |
|---|---|---|
| **RLDS**（TFRecord） | OpenVLA、Octo、Open X-Embodiment | Google 系，按 episode 存储 |
| **LeRobotDataset**（parquet + mp4） | LeRobot、SmolVLA、openpi 微调 | HuggingFace 系，图像存成视频，按帧索引 |

参考源码：`third_party/lerobot/src/lerobot/datasets/lerobot_dataset.py`，注意 `__init__` 里的 `delta_timestamps` 参数：它的作用就是“取当前帧时，顺便把未来 H 帧的动作也取出来”，也就是下面要说的 action chunk。

### 1.2 预处理流水线（训练和推理都要走一遍）

```
原始帧 ──► ①图像: resize(224) / 归一化 / 数据增强(随机裁剪、颜色抖动)
       ──► ②语言: 套 prompt 模板 → tokenizer → ids + mask
       ──► ③状态/动作: 归一化 (mean-std 或 分位数 q01/q99 → [-1,1])
       ──► ④动作分块: actions[t : t+H] + is_pad 掩码
       ──► ⑤padding: 把不同机器人的动作维度补齐到统一维度（π0 是 32 维）
```

**为什么归一化这么重要？** 不同机器人的动作量纲完全不同（米、弧度、0/1 夹爪）。OpenVLA 用 1%/99% 分位数把每一维映射到 [-1,1]，推理时再用**该数据集自己的**统计量反归一化（这就是 OpenVLA 里 `unnorm_key` 参数的意义）。见 `third_party/openvla/prismatic/extern/hf/modeling_prismatic.py` 的 `predict_action`：

```python
actions = np.where(mask, 0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low, normalized_actions)
```

`common.py` 里的 `QuantileNormalizer` 就是它的复刻。

**Action chunk 是什么？** 每个时刻不只预测下一步动作，而是预测未来 H 步（ACT 论文提出；π0 中 H=50，SmolVLA 中 H=50）。好处有两个：
- 推理一次可以执行好几步，**弥补大模型推理慢的问题**
- 动作更平滑，减少逐步决策时误差累积（compounding error）

episode 结尾不够 H 步的地方补 0，并用 `is_pad` 在 loss 里 mask 掉。跑 `00_explore_data.py` 可以看到：

```
episode 0 最后一帧的 is_pad: [0, 1, 1, 1, 1, 1, 1, 1]
```

---

## 2. 通用架构骨架

所有主流 VLA 都可以拆成这 4 块：

```
              ┌────────────────────────────── VLM（从互联网图文预训练继承）─────────────────────────────┐
 相机图像 ───►│ ① Vision Encoder ──► ② Projector ──┐                                                    │
             │   (SigLIP/DINOv2 ViT)   (MLP)      ├──► ③ LLM Backbone (Llama-2 7B / Gemma 2B / SmolLM) │
 语言指令 ───►│   tokenizer ──► embedding ─────────┘         (Transformer)                             │
             └─────────────────────────────────────────────────────┬──────────────────────────────────┘
 机器人状态 ─────────────────────────────────────────────────────┐   │ hidden states / KV
                                                                 ▼   ▼
                                                   ④ Action Head（这里是各家分歧最大的地方）
                                                        │
                                                        ▼
                                          反归一化 → 动作块 a[t:t+H] → 机器人控制器
```

①②③ 就是标准的 LLaVA 式 VLM，你应该很熟悉：ViT 把图像切成 patch 得到 N 个视觉 token，MLP 投影到 LLM 的隐藏维度，再和文本 token 拼接后送进 LLM。

**④ 动作头**是理解 VLA 的关键，主要分两条路线：

| 路线 | 代表模型 | 动作表示 | 动作头 | Loss |
|---|---|---|---|---|
| **A. 离散 token（自回归）** | RT-2、OpenVLA、π0-FAST | 每维分 256 个 bin，或 FAST（DCT+BPE）压缩 | 就是 LLM 自己的 LM head | 交叉熵 |
| **B. 连续生成（扩散 / 流匹配）** | π0、π0.5、SmolVLA、GR00T N1、Octo、Diffusion Policy | 连续的动作块 (H, D) | 单独的 Action Expert / DiT | flow matching / 扩散去噪 MSE |
| （C. 直接回归） | ACT、OpenVLA-OFT | 连续动作块 | MLP 或 L1 回归头 | L1 / MSE |

下面两课分别用代码实现 A 和 B。

---

## 3. 路线 A：动作即 token（第 2 课，`01_mini_openvla.py`）

### 3.1 思想

既然 LLM 只会预测下一个 token，那就**把动作也变成 token**：

```
连续动作 (dx,dy) = (-0.0777, -0.0134)
  └─ 归一化到 [-1,1]              → (-0.843, -0.148)
  └─ 每维均匀切 256 个 bin        → bin 索引
  └─ 映射到词表最后 256 个 id     → token [252, 164]     ← OpenVLA 直接覆盖 Llama 词表里最少用的 256 个 token
```

训练时把样本组织成一段“对话”：

```
[图像 patch ×64] [BOS] go to the red block [SEP] [a1] [a2]
                 └──────────── prompt ─────────┘ └ 只在这里算 loss ┘
```

OpenVLA 真实的 prompt 是 `"In: What action should the robot take to {instruction}?\nOut: "` 后面接 7 个动作 token（7 维动作）。

### 3.2 对照源码

| 步骤 | 迷你代码 | OpenVLA 源码 |
|---|---|---|
| 动作 ↔ token | `ActionTokenizer` | `third_party/openvla/prismatic/vla/action_tokenizer.py` |
| 构造 input_ids / labels | `build_batch` | `third_party/openvla/prismatic/vla/datasets/datasets.py` → `RLDSBatchTransform` |
| 视觉编码 | `TinyViT` | `modeling_prismatic.py` → `PrismaticVisionBackbone`（DINOv2 + SigLIP 双塔，按通道拼接） |
| 投影 | `projector` (Linear-GELU-Linear) | `modeling_prismatic.py` → `PrismaticProjector` |
| LLM | 4 层因果 Transformer | Llama-2 7B |
| 推理 | `predict_action` | `modeling_prismatic.py` → `predict_action`（`generate` 出 7 个 token → bin center → 反归一化） |

源码里最关键的一行（`datasets.py`）：

```python
# [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
labels[: -(len(action) + 1)] = IGNORE_INDEX
```

### 3.3 跑一下看数据流

```
[3] 一次前向的张量形状:
    image          (4, 64, 64, 3)
    visual tokens  (4, 64, 192)   (ViT patch -> projector)
    text+action ids(4, 12) -> embeds (4, 12, 192)       # 1 BOS + 8 词 + 1 SEP + 2 动作
    LLM input      (4, 76, 192)   attn mask (4, 76, 76)
    logits         (4, 76, 273)                         # 17 个文本词 + 256 个动作 bin
    labels[0] = [-100 × 10, 252, 164]
```

把它换成真实的 OpenVLA，形状就是：image (1,6,224,224)（两个 ViT 各 3 通道）→ 256 个 patch token × 4096 维 → 加上约 20 个文本 token → Llama-2 → 自回归生成 7 个 token。

### 3.4 优缺点

- ✅ 完全复用 LLM 的训练和推理基础设施，能和图文数据联合训练（RT-2 的 co-training）
- ❌ **慢**：7 维动作要跑 7 次自回归解码；如果要输出动作块（H×D 个 token）就更慢
- ❌ 精度受 bin 数量限制；各个维度之间是串行生成的
- 改进：**FAST tokenizer**（π0-FAST）先对动作块做 DCT，再用 BPE 压缩，token 数大大减少；**OpenVLA-OFT** 改成并行解码 + L1 回归，推理快 25 倍以上

---

## 4. 路线 B：Flow Matching + Action Expert（第 3 课，`02_mini_pi0.py`）

这是目前（2025–2026 年）最主流的设计：π0 / π0.5 / SmolVLA / GR00T 都属于这条路线。

### 4.1 为什么要用生成模型做动作头

示教数据里同一个场景可能有多种走法（多模态分布）。用 MSE 直接回归会学到“平均动作”。Diffusion / Flow Matching 学的是**整个分布**，采样出来的是其中某一种合理的走法。

### 4.2 Flow Matching 30 秒速成

把“从噪声到动作”看成一条直线路径，τ∈[0,1]（π0 的约定：**τ=1 是纯噪声，τ=0 是真实动作**）：

$$x_\tau = \tau\,\epsilon + (1-\tau)\,a,\qquad \epsilon\sim\mathcal N(0,I)$$

这条路径的速度（对 τ 求导）恒定为 $u = \epsilon - a$。让网络预测这个速度：

$$\mathcal L = \mathbb E_{\tau,\epsilon}\;\big\|\,v_\theta(x_\tau,\tau\mid \text{obs}) - (\epsilon - a)\,\big\|^2$$

推理时从纯噪声出发，用欧拉法积分 10 步：$x \leftarrow x - \tfrac{1}{10}\,v_\theta(x,\tau)$，τ 从 1 走到 0，就得到动作块。

对照 `third_party/openpi/src/openpi/models/pi0.py` → `compute_loss`，代码几乎和公式一模一样：

```python
time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001   # τ 偏向高噪声端
x_t = time_expanded * noise + (1 - time_expanded) * actions
u_t = noise - actions
...
v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
return jnp.mean(jnp.square(v_t - u_t), axis=-1)
```

### 4.3 架构：prefix / suffix + 双专家共享注意力

π0 把序列分成两段：

```
prefix（VLM 权重：PaliGemma 2B, width 2048）       suffix（Action Expert 权重：Gemma 300M, width 1024）
[img_1 ... img_N  |  text_1 ... text_M]            [state | noisy_act_1 ... noisy_act_H (+ τ 编码)]
      embed_prefix()                                       embed_suffix()
```

关键设计：**Mixture-of-Transformers**。每一层里两段 token 用**各自的权重**（QKV 投影、MLP、LayerNorm 都分开），但 attention 是**拼在一起算的**。所以动作 token 在每一层都能直接“看到”VLM 对图像和语言的理解。

迷你代码 `MoTLayer.forward` 就是这个结构：

```python
for i, x in enumerate(xs):                 # xs = [prefix, suffix]
    q, k, v = self.qkv[i](self.norm1[i](x))   # 各用各的权重
k, v, q = torch.cat(ks, 2), torch.cat(vs, 2), torch.cat(qs, 2)
h = scaled_dot_product_attention(q, k, v, mask)   # 一次联合 attention
# 再按 token 归属切回去，分别走自己的 out-proj 和 MLP
```

### 4.4 Attention Mask：VLA 里最值得仔细看的东西

openpi 用 `ar_mask` 的累加和来定义“块因果”注意力（`pi0.py` → `make_attn_mask`）：token i 能看到 token j，当且仅当 `cumsum[j] <= cumsum[i]`。

```
ar_mask:  img:0 0 0 ... text:0 0 0 | state:1 | act:1 0 0 0 ...
cumsum :      0 0 0 ...      0 0 0 |       1 |     2 2 2 2 ...
```

跑 `02_mini_pi0.py` 会打印出来：

```
              img    text   state  action
     img       ●       ◐       ○       ○      ← prefix 看不到 suffix
    text       ◐       ◐       ○       ○      （◐ 是因为文本有 padding）
   state       ●       ◐       ●       ○
  action       ●       ◐       ●       ●      ← 动作块内部双向，可以看到全部内容
```

为什么要这样设计？
- **prefix 看不到 suffix** → prefix 的计算结果和噪声动作无关 → 推理时 **prefix 只需算一次并缓存 KV**，10 步去噪每一步只需要跑短短的 suffix。这是 π0 能做到实时控制的关键（见 `sample_actions` 里先填 `kv_cache` 再进 `while_loop`）
- **prefix 内部双向**：图像和语言之间互相看（prefix-LM），和 PaliGemma 预训练时保持一致
- **动作块内部双向**：动作块是整体去噪的，不是自回归

### 4.5 推理时的数据流

```
obs ─► embed_prefix ─► 跑一遍全部层，缓存每层 prefix 的 (K,V)          ← 1 次，开销大
x ~ N(0, I)  (H, D)
for k in 0..9:                                                         ← 10 次，开销小
    τ = 1 - k/10
    embed_suffix(state, x, τ) ─► 每层: suffix 的 Q 去 attend [缓存的 prefix KV | suffix KV]
    v = action_out_proj(...)
    x = x - 0.1 * v
反归一化(x) ─► 动作块 (H, D) ─► 执行前 n 步 ─► 新观测 ─► 重新规划
```

迷你代码 `MiniPi0.sample_actions` 完整实现了这个流程（包括 KV cache）。

### 4.6 π0 / π0.5 / SmolVLA 的差异

| | π0 | π0.5 | SmolVLA（LeRobot） |
|---|---|---|---|
| VLM | PaliGemma 3B | PaliGemma | SmolVLM2 500M（只用前 16 层） |
| 动作专家 | Gemma 300M | 同左 | 宽度为 VLM 的 0.75 倍 |
| state 放在哪 | suffix 里的一个 token | **离散化成文本**放进 prefix | prefix 末尾 |
| τ 怎么注入 | 和动作拼接后过 MLP | **adaRMSNorm** 调制每一层 | 和动作拼接后过 MLP |
| 注意力 | 联合 self-attention | 同左 | **交替使用 cross-attn / self-attn** |
| 源码 | `openpi/src/openpi/models/pi0.py` | 同文件里的 `self.pi05` 分支 | `lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py` |

建议：读完迷你代码后，打开 SmolVLA 的 `embed_prefix` / `embed_suffix` / `forward`（`modeling_smolvla.py` 第 591–762 行）对照着看，你会发现结构几乎一模一样，只是规模更大。它的 `make_att_2d_masks` 在 `lerobot/src/lerobot/policies/common/vla_utils.py`，注释写着 "see openpi (exact copy)"。

---

## 5. 端到端数据流总结（真实 π0 部署时）

以 openpi 在 LIBERO 仿真环境上的推理为例（`openpi/src/openpi/policies/policy_config.py` → `create_trained_policy`）：

```
机器人/仿真 obs dict
  {"observation/image", "observation/wrist_image", "observation/state", "prompt"}
     │ ① repack_transforms         ：改键名
     │ ② data_transforms (LiberoInputs, libero_policy.py)
     │                             ：整理成 {"image": {base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb(全零+mask)}, "state", "prompt"}
     │ ③ Normalize(norm_stats)     ：state 归一化
     │ ④ model_transforms          ：ResizeImages(224) / TokenizePrompt / PadStatesAndActions(→32维)
     ▼
  Observation ──► Pi0.sample_actions()  (prefix KV cache + 10 步 flow matching)
     ▼
  actions (50, 32)
     │ ⑤ model_transforms.outputs
     │ ⑥ Unnormalize(norm_stats)   ：反归一化
     │ ⑦ data_transforms.outputs (LiberoOutputs)：截取前 7 维
     ▼
  (50, 7) 动作块 → 客户端执行前若干步（action chunking）→ 采集新观测 → 再次请求
```

**训练和推理必须走完全相同的变换和归一化统计量**，这是新手部署 VLA 时最常踩的坑。

---

## 6. 打通流程后的动手练习

每个脚本的打印输出都对应上文的一个环节：`[1]` 数据 → `[2]` 动作表示 → `[3]` 一次前向的张量形状 / mask → `[4]` 训练 loss → `[5]` 推理与闭环执行。建议先对照输出把第 3、4 节的数据流图在脑子里走一遍，再改代码做练习。

动手练习（建议按顺序做）：

0. **想看模型真正学会**：加大参数，例如 `--episodes 10000 --steps 4000`（单卡大约 8 分钟）。参考：mini-π0 在这个设置下成功率可以达到很高；mini-OpenVLA 在同样设置下明显更难训练（256 个 bin 的交叉熵收敛慢），这本身就是路线 A 的一个特点，值得思考原因
1. **验证模型用到了语言**：在评测时把指令里的颜色换掉（例如 `obs["instruction"] = "go to the green block"`），看机械臂是否走向另一个方块
2. **bin 数量的影响**：`01_mini_openvla.py --bins 16`，成功率如何变化？为什么？
3. **action chunking**：把 `02_mini_pi0.py --execute` 设为 1 / 4 / 8，比较成功率和推理次数
4. **去噪步数**：把 `sample_actions(num_steps=10)` 改成 1、3、20，体会 flow matching 少步采样的能力
5. **注意力 mask**：把 `embed_suffix` 中 action 的 `ar` 全部改成 True（动作块内部变成因果），会怎样？
6. **改成纯回归**：把 flow matching 换成直接 MSE 回归动作块（相当于 ACT / OFT），比较效果。如果把专家改成“随机从左或从右绕”（多模态数据），回归会出现什么问题？

---

## 7. 下一步：上手真实模型

1. **读论文**（按顺序）：RT-2 → OpenVLA → ACT（action chunking）→ Diffusion Policy → π0 → FAST → π0.5 → SmolVLA
2. **读源码**（已经在 `third_party/` 下）：
   - OpenVLA：`prismatic/extern/hf/modeling_prismatic.py`、`prismatic/vla/action_tokenizer.py`、`prismatic/vla/datasets/datasets.py`
   - openpi：`src/openpi/models/pi0.py`、`src/openpi/models/gemma.py`（多专家 Transformer）、`src/openpi/policies/libero_policy.py`、`src/openpi/transforms.py`
   - LeRobot：`src/lerobot/policies/smolvla/`、`src/lerobot/policies/pi0/`、`src/lerobot/datasets/lerobot_dataset.py`
3. **跑真实模型**：SmolVLA（450M）最适合入门，可以在 LeRobot 中直接加载 `lerobot/smolvla_base`，在 LIBERO 或 SO-100 数据集上微调。需要的话我可以在 `vla` conda 环境里帮你装好 LeRobot，并下载权重跑通推理。
