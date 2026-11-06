"""教程用的示例模型。都很小，几秒内就能跑完，但覆盖了常见的性能现象。

- TinyCNN       : 类 ResNet 的卷积网络，计算密集；也是第 06 章 TensorRT 的导出对象
- TinyTransformer: 标准 Encoder，覆盖 Attention / LayerNorm / GEMM
- ManySmallOps  : 一串很小的逐元素算子，专门制造 Launch-bound
- ToyVLA        : 视觉编码 + 语言 token + Transformer + 动作头，近似 VLA 的结构（第 08 章使用）
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.short = (nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
                      if stride != 1 or cin != cout else nn.Identity())

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(y + self.short(x))


class TinyCNN(nn.Module):
    """输入 (B,3,224,224)。features() 输出 (B, C, 7, 7)，forward() 输出分类 logits。"""

    def __init__(self, num_classes: int = 1000, width: int = 64):
        super().__init__()
        w = width
        self.stem = nn.Sequential(nn.Conv2d(3, w, 7, 2, 3, bias=False), nn.BatchNorm2d(w), nn.ReLU(),
                                  nn.MaxPool2d(3, 2, 1))
        self.layers = nn.Sequential(ConvBlock(w, w), ConvBlock(w, 2 * w, 2),
                                    ConvBlock(2 * w, 4 * w, 2), ConvBlock(4 * w, 8 * w, 2))
        self.out_channels = 8 * w
        self.fc = nn.Linear(8 * w, num_classes)

    def features(self, x):
        return self.layers(self.stem(x))

    def forward(self, x):
        return self.fc(torch.flatten(F.adaptive_avg_pool2d(self.features(x), 1), 1))


class TinyTransformer(nn.Module):
    """输入 (B,T,d_model)，输出同形状。"""

    def __init__(self, d_model: int = 512, nhead: int = 8, num_layers: int = 4, dim_ff: int = 2048):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout=0.0, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(self.encoder(x))


class ManySmallOps(nn.Module):
    """n_ops 个很小的逐元素算子串联。每个算子在 GPU 上只跑几微秒，launch 开销占大头。"""

    def __init__(self, n_ops: int = 200):
        super().__init__()
        self.n_ops = n_ops

    def forward(self, x):
        for i in range(self.n_ops):
            x = x * 1.0001 + 0.0001 if i % 2 == 0 else torch.relu(x)
        return x


class ToyVLA(nn.Module):
    """简化版 VLA：image (B,3,224,224) + tokens (B,L) -> actions (B, chunk, action_dim)。

    视觉 token(49) 和语言 token(L) 拼接后送入 Transformer，再从 learnable action query 读出动作。
    """

    def __init__(self, d_model: int = 512, vocab: int = 32000, n_layers: int = 6, chunk: int = 16,
                 action_dim: int = 7):
        super().__init__()
        self.vision = TinyCNN(width=64)
        self.vis_proj = nn.Linear(self.vision.out_channels, d_model)
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.action_query = nn.Parameter(torch.randn(1, chunk, d_model) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, 256, d_model) * 0.02)
        self.backbone = TinyTransformer(d_model, 8, n_layers, 4 * d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, action_dim))
        self.chunk = chunk

    def forward(self, image, tokens):
        v = self.vision.features(image)                       # (B,C,7,7)
        v = self.vis_proj(v.flatten(2).transpose(1, 2))       # (B,49,d)
        t = self.tok_emb(tokens)                              # (B,L,d)
        q = self.action_query.expand(image.shape[0], -1, -1)  # (B,chunk,d)
        x = torch.cat([v, t, q], dim=1)
        x = x + self.pos[:, : x.shape[1]]
        x = self.backbone(x)
        return torch.tanh(self.head(x[:, -self.chunk:]))      # (B,chunk,action_dim) 归一化到 [-1,1]

