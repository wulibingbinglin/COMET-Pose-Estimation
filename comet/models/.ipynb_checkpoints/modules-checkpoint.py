# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Callable
import collections
from torch import Tensor
from itertools import repeat


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


to_2tuple = _ntuple(2)


class ResidualBlock(nn.Module):
    """
    ResidualBlock: construct a block of two conv layers with residual connections
    """

    def __init__(
        self, in_planes, planes, norm_fn="group", stride=1, kernel_size=3
    ):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=kernel_size,
            padding=1,
            stride=stride,
            padding_mode="zeros",
        )
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=kernel_size,
            padding=1,
            padding_mode="zeros",
        )
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == "group":
            self.norm1 = nn.GroupNorm(
                num_groups=num_groups, num_channels=planes
            )
            self.norm2 = nn.GroupNorm(
                num_groups=num_groups, num_channels=planes
            )
            if not stride == 1:
                self.norm3 = nn.GroupNorm(
                    num_groups=num_groups, num_channels=planes
                )

        elif norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == "none":
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()
        else:
            raise NotImplementedError

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride),
                self.norm3,
            )

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = (
            partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear
        )

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Attention(nn.Module):
    def __init__(
        self, query_dim, context_dim=None, num_heads=8, dim_head=48, qkv_bias=False # query_dim = hidden_size = 384
    ):
        super().__init__()
        inner_dim = dim_head * num_heads # 每个头的维度乘以头的数量，表示所有头的输出维度大小 =384
        context_dim = default(context_dim, query_dim) # 上下文（K 和 V）向量的维度。如果没有提供，则默认为 query_dim
        self.scale = dim_head**-0.5 # 缩放因子，dim_head**-0.5 是 Transformer 中常见的做法，用于缩放相似度计算，避免数值过大。
        self.heads = num_heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=qkv_bias) # 从查询向量到多头输出的线性变换。
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=qkv_bias) # 从上下文向量（键和值）到多头输出的线性变换。context_dim 乘以 2（键和值）是因为需要分别计算 K 和 V。
        self.to_out = nn.Linear(inner_dim, query_dim) # 最后的线性变换，将所有头的输出映射回查询维度。

    def forward(self, x, context=None, attn_bias=None):
        # 就是简单的多头注意力block context.shape = B, N2, context_dim = 实际也等于hidden_size
        B, N1, C = x.shape # transformer的要求，三维，第一为batchsize，第二为序列长度，第三为每个token的维度  C = query_dim =384
        h = self.heads

        q = self.to_q(x).reshape(B, N1, h, -1).permute(0, 2, 1, 3) # 三维变四维
        context = default(context, x) #因为有时候，我们的key和value是来源于别的，就是利用的是cross-attention
        k, v = self.to_kv(context).chunk(2, dim=-1) # chunk(2, dim=-1) 将输出分成两部分，分别表示键和值 （三维）

        N2 = context.shape[1]
        k = k.reshape(B, N2, h, C // h).permute(0, 2, 1, 3)
        v = v.reshape(B, N2, h, C // h).permute(0, 2, 1, 3) # [batch, heads, seq_len, head_dim]

        sim = (q @ k.transpose(-2, -1)) * self.scale # (B, h, N1, N2) 计算查询和键的点积相似度（即 $Q \cdot K^T$），然后乘以缩放因子 scale，以避免在训练过程中梯度爆炸问题。

        if attn_bias is not None:
            sim = sim + attn_bias
        # 通常用于掩蔽（masking）无效的位置，例如在计算自注意力时，遮蔽掉填充位置（padding），或者进行因果遮蔽（causal masking），确保模型不会“看到”未来的信息
        # attn_bias 是一个可选的矩阵，通常在 sim（即查询和键的相似度矩阵）上进行加法操作，作为一个 偏置 或 掩蔽，用于调整注意力的计算结果。
        # 这个偏置矩阵 attn_bias 通常包含的是负的极大值（例如 -1e9），目的是将某些无效的或被掩蔽的位置的相似度值调整为极小值，使得这些位置在 softmax 操作后变得 几乎为零。
        attn = sim.softmax(dim=-1) #

        x = (attn @ v).transpose(1, 2).reshape(B, N1, C)  # (B, h, N1, C) -》(B, N1, h, C)-> (B, N1, C) x：计算加权和值（即通过注意力权重对值进行加权）qurey的序列数决定了最终输出的序列数,
        return self.to_out(x)


class AttnBlock_2(nn.Module):
    def __init__(
        self,
        hidden_size, #所有头的维度
        num_heads,
        attn_class: Callable[..., nn.Module] = Attention,
        mlp_ratio=4.0,  # MLP隐藏层的大小与输入 hidden_size 的比例，默认值为 4.0。
        **block_kwargs
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # self.norm1 是第一个 LayerNorm 层，它用于对输入进行规范化，帮助加速训练并稳定训练过程。
        self.attn = attn_class(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs
        )
        # hidden_size 是输入的维度,qkv_bias=True 表示在计算查询、键和值时使用偏置项。

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # self.norm2 是第二个 LayerNorm 层，它作用于 MLP 输出的结果。
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # 计算 MLP 隐藏层的维度，mlp_ratio 是默认设置为 4.0，表示 MLP 隐藏层维度是输入维度的 4 倍。
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        # 这是一个近似 GELU 激活函数的定义，使用 tanh 近似。
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        # self.mlp 是一个多层感知机（MLP）模块。

    def forward(self, x, mask=None):
        """ 这个 AttnBlock 类是一个典型的基于 自注意力机制 和 MLP（多层感知机）的 Transformer 模块，实际上就是VIT的一个block"""
        attn_bias = mask #         attn_bias = mask
        if mask is not None:
            mask = (
                (mask[:, None] * mask[:, :, None])
                # 现在，我们来对 mask[:, None] 和 mask[:, :, None] 进行相乘，形状分别是 (2, 1, 3) 和 (2, 3, 1)。PyTorch 会自动广播它们，使得它们可以进行逐元素相乘。
                # 具体来说，广播机制会使得它们的维度对齐如下：
                # mask[:, None] 的形状 (2, 1, 3) 会被广播成 (2, 3, 3)（在第二维上复制 3 次）。
                # mask[:, :, None] 的形状 (2, 3, 1) 会被广播成 (2, 3, 3)（在第三维上复制 3 次）
                .unsqueeze(1) # 接下来，通过 unsqueeze(1)，我们在第二维插入一个新的维度。最终得到一个形状为 [batch_size, 1, seq_len, seq_len] 的张量：
                .expand(-1, self.attn.num_heads, -1, -1) # 使用 expand 将这个张量扩展到多个头部（num_heads）
            )
            max_neg_value = -torch.finfo(x.dtype).max
            attn_bias = (~mask) * max_neg_value
        x = x + self.attn(self.norm1(x), attn_bias=attn_bias)
        x = x + self.mlp(self.norm2(x))
        return x


class AttnBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        attn_class: Callable[..., nn.Module] = nn.MultiheadAttention,
        mlp_ratio=4.0,
        **block_kwargs
    ):
        """
        Self attention block
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )

        self.attn = attn_class(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True, # 竟然偷偷加这个
            **block_kwargs
        )

        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=mlp_hidden_dim, drop=0
        )

    def forward(self, x, mask=None):
        # Prepare the mask for PyTorch's attention (it expects a different format)
        # attn_mask = mask if mask is not None else None
        # Normalize before attention
        x = self.norm1(x)

        # PyTorch's MultiheadAttention returns attn_output, attn_output_weights
        # attn_output, _ = self.attn(x, x, x, attn_mask=attn_mask)

        attn_output, _ = self.attn(x, x, x)

        # Add & Norm
        x = x + attn_output
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttnBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        context_dim,
        num_heads=1,
        mlp_ratio=4.0,
        **block_kwargs
    ):
        """
        Cross attention block
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.norm_context = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True,
            **block_kwargs
        )

        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=mlp_hidden_dim, drop=0
        )

    def forward(self, x, context, mask=None):
        # Normalize inputs
        x = self.norm1(x)
        context = self.norm_context(context)

        # Apply cross attention
        # Note: nn.MultiheadAttention returns attn_output, attn_output_weights
        attn_output, _ = self.cross_attn(x, context, context, attn_mask=mask)

        # Add & Norm
        x = x + attn_output
        x = x + self.mlp(self.norm2(x))
        return x
