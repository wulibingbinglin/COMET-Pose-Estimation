# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# Modified from https://github.com/facebookresearch/co-tracker/


import torch # 用于构建神经网络的模块。
import torch.nn as nn
import torch.nn.functional as F # 包含一些函数式的操作（如激活函数等）。
from functools import partial
from typing import Callable
import collections
from torch import Tensor
from itertools import repeat

from torch.nn.init import trunc_normal_

from ..utils import bilinear_sampler

from ..modules import Mlp, AttnBlock, CrossAttnBlock, ResidualBlock, AttnBlock_2


class BasicEncoder(nn.Module): # BasicEncoder 继承自 torch.nn.Module，是一个卷积神经网络的编码器。
    def __init__(self, input_dim=3, output_dim=128, stride=4, use_trans = False, cfg=None):
        # 输入图像的通道数，默认是 3（表示彩色图像 RGB）
        # 输出的特征维度，默认是 128
        super(BasicEncoder, self).__init__()

        self.stride = stride
        self.norm_fn = "instance" # 指定使用的归一化类型，这里是 instance，即 实例归一化（Instance Normalization）。
        self.in_planes = output_dim // 2 # 根据输出维度设置输入通道数的一半，用于后面的层。

        self.norm1 = nn.InstanceNorm2d(self.in_planes)
        self.norm2 = nn.InstanceNorm2d(output_dim * 2)
        # 实例归一化层。nn.InstanceNorm2d 是一个对 2D 输入（例如图像）的实例归一化层。它会对每个样本单独进行归一化

        self.conv1 = nn.Conv2d( # 512，512-》(256,256)
            input_dim, # 输入通道数，表示输入是 3 通道的图像
            self.in_planes, # 卷积输出的通道数
            kernel_size=7, # 卷积核大小为 7x7
            stride=2, # 步幅为 2，意味着卷积时每次跳过 2 个像素
            padding=3, # 为了保持卷积后输出的大小，使用 3 像素的零填充
            padding_mode="zeros", # 填充方式为零填充
        ) # 这只是初始化了一个卷积层的结构
        # 输出通道数决定了卷积核的数量，每个卷积核会学习一种特定的特征
        self.relu1 = nn.ReLU(inplace=True) # ReLU 激活函数，用于对卷积层的输出进行非线性变换, 表示在原地进行操作，节省内存
        self.layer1 = self._make_layer(output_dim // 2, stride=1) # stride 导致大小不变
        self.layer2 = self._make_layer(output_dim // 4 * 3, stride=2) # stride 会导致大小减半
        self.layer3 = self._make_layer(output_dim, stride=2) # stride 会导致大小减半
        self.layer4 = self._make_layer(output_dim, stride=2) # stride 会导致大小减半
        # 这些是网络的多个层，分别调用了 _make_layer 函数来构建。每个层的输入输出通道数和步幅设置不同。

        self.conv2 = nn.Conv2d(
            output_dim * 3 + output_dim // 4,
            output_dim * 2,
            kernel_size=3,
            padding=1,
            padding_mode="zeros",
        )
        # 另一个卷积层，将上一层的输出通道数和 output_dim // 4 的输出拼接在一起，得到 output_dim * 3 + output_dim // 4 的输入通道。
        self.relu2 = nn.ReLU(inplace=True) # 另一个 ReLU 激活层。
        self.conv3 = nn.Conv2d(output_dim * 2, output_dim, kernel_size=1)
        # 一个 1x1 卷积层，将输入通道数从 output_dim * 2 缩减到 output_dim。

        for m in self.modules(): # 遍历网络中的所有子模块。
            if isinstance(m, nn.Conv2d): # 检查当前模块是否是卷积层。
                nn.init.kaiming_normal_( # 使用 Kaiming 初始化（也叫 He 初始化）来初始化卷积层的权重，适用于 ReLU 激活函数。
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, (nn.InstanceNorm2d)): # 检查当前模块是否是实例归一化层,：将归一化层的权重初始化为 1，偏置初始化为 0。
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1): # 通常它是内部方法（由下划线 _ 开头），表示它不应该被直接调用，而是供其他方法或类内部使用。
        # 表示该层的输出通道数（特征的数量）
        """_make_layer 方法的作用是定义和返回一个包含多个残差块（ResidualBlock）的神经网络层。"""
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        _, _, H, W = x.shape # 512 512 因为down_ratio搞到了1/2

        x = self.conv1(x) # 8 64 256 256 x输入的大小/2
        x = self.norm1(x)
        x = self.relu1(x)

        a = self.layer1(x) # 8 64 256 256
        b = self.layer2(a) # 8 96 128 128
        c = self.layer3(b) # 8 128 64 64
        d = self.layer4(c) # 8 128 32 32

        a = _bilinear_intepolate(a, self.stride, H, W) # 8 64 128 128 原大小/8
        b = _bilinear_intepolate(b, self.stride, H, W) # 8 96 128 128 1024/8
        c = _bilinear_intepolate(c, self.stride, H, W) # 8 128 128 128
        d = _bilinear_intepolate(d, self.stride, H, W) # 8 128 128 128

        x = self.conv2(torch.cat([a, b, c, d], dim=1)) # 8 256 128 128
        x = self.norm2(x) #
        x = self.relu2(x)
        x = self.conv3(x) # 8 128 原大小/8=128 128
        return x


class ShallowEncoder(nn.Module): # 精细跟踪的特征提取器
    def __init__(
        self, input_dim=3, output_dim=32, stride=1, norm_fn="instance", cfg=None
    ):# 若开启融合策略，那inputdim就变了3+32=35
        super(ShallowEncoder, self).__init__()
        self.stride = stride
        self.norm_fn = norm_fn
        self.in_planes = output_dim #32


        if self.norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=self.in_planes)
            self.norm2 = nn.GroupNorm(num_groups=8, num_channels=output_dim * 2)
        elif self.norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(self.in_planes)
            self.norm2 = nn.BatchNorm2d(output_dim * 2)
        elif self.norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(self.in_planes)
            self.norm2 = nn.InstanceNorm2d(output_dim * 2)
        elif self.norm_fn == "none":
            self.norm1 = nn.Sequential()

        # 看似好像比basicencoder还要短，从第二个开始的
        self.conv1 = nn.Conv2d(
            input_dim, # 3
            self.in_planes, #32
            kernel_size=3,
            stride=2,
            padding=1,
            padding_mode="zeros",
        )
        self.relu1 = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(output_dim, stride=2)

        self.layer2 = self._make_layer(output_dim, stride=2)
        self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(
                m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)
            ):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        self.in_planes = dim

        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        return layer1

    def forward(self, x):
        _, _, H, W = x.shape

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        tmp = self.layer1(x)
        x = x + F.interpolate(
            tmp, (x.shape[-2:]), mode="bilinear", align_corners=True
        )
        tmp = self.layer2(tmp)
        x = x + F.interpolate(
            tmp, (x.shape[-2:]), mode="bilinear", align_corners=True
        )
        tmp = None
        x = self.conv2(x) + x

        x = F.interpolate(
            x,
            (H // self.stride, W // self.stride),
            mode="bilinear",
            align_corners=True,
        )

        return x


def _bilinear_intepolate(x, stride, H, W):
    return F.interpolate(
        x, (H // stride, W // stride), mode="bilinear", align_corners=True
    ) # 插值成对应的大小 (H // stride, W // stride)


class EfficientUpdateFormer(nn.Module):
    """
    Transformer model that updates track estimates.
    """

    def __init__(
        self,
        space_depth=6,
        time_depth=6,
        input_dim=320,
        hidden_size=384,
        num_heads=8,
        output_dim=130,
        mlp_ratio=4.0,
        add_space_attn=True,
        num_virtual_tracks=64, #固定的
    ):
        super().__init__()

        self.out_channels = 2
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.add_space_attn = add_space_attn
        self.input_transform = torch.nn.Linear(
            input_dim, hidden_size, bias=True
        )
        self.flow_head = torch.nn.Linear(hidden_size, output_dim, bias=True)
        self.num_virtual_tracks = num_virtual_tracks

        if self.add_space_attn:
            self.virual_tracks = nn.Parameter(
                torch.randn(1, num_virtual_tracks, 1, hidden_size)
            )
        else:
            self.virual_tracks = None

        self.time_blocks = nn.ModuleList(
            [
                AttnBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_class=nn.MultiheadAttention,
                )
                for _ in range(time_depth)
            ]
        )

        if add_space_attn:
            self.space_virtual_blocks = nn.ModuleList(
                [
                    AttnBlock(
                        hidden_size,
                        num_heads,
                        mlp_ratio=mlp_ratio,
                        attn_class=nn.MultiheadAttention,
                    )
                    for _ in range(space_depth)
                ]
            )
            self.space_point2virtual_blocks = nn.ModuleList(
                [
                    CrossAttnBlock(
                        hidden_size, hidden_size, num_heads, mlp_ratio=mlp_ratio
                    )
                    for _ in range(space_depth)
                ]
            )
            self.space_virtual2point_blocks = nn.ModuleList(
                [
                    CrossAttnBlock(
                        hidden_size, hidden_size, num_heads, mlp_ratio=mlp_ratio
                    )
                    for _ in range(space_depth)
                ]
            )
            assert len(self.time_blocks) >= len(self.space_virtual2point_blocks)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        def init_weights_vit_timm(module: nn.Module, name: str = ""):
            """ViT weight initialization, original timm impl (for reproducibility)"""
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_tensor, mask=None):
        tokens = self.input_transform(input_tensor)

        init_tokens = tokens

        B, _, T, _ = tokens.shape

        if self.add_space_attn:
            virtual_tokens = self.virual_tracks.repeat(B, 1, T, 1)
            tokens = torch.cat([tokens, virtual_tokens], dim=1)

        _, N, _, _ = tokens.shape

        j = 0
        for i in range(len(self.time_blocks)):
            time_tokens = tokens.contiguous().view(
                B * N, T, -1
            )  # B N T C -> (B N) T C
            time_tokens = self.time_blocks[i](time_tokens)

            tokens = time_tokens.view(B, N, T, -1)  # (B N) T C -> B N T C
            if self.add_space_attn and (
                i % (len(self.time_blocks) // len(self.space_virtual_blocks))
                == 0
            ):
                space_tokens = (
                    tokens.permute(0, 2, 1, 3).contiguous().view(B * T, N, -1)
                )  # B N T C -> (B T) N C
                point_tokens = space_tokens[:, : N - self.num_virtual_tracks]
                virtual_tokens = space_tokens[:, N - self.num_virtual_tracks :]

                virtual_tokens = self.space_virtual2point_blocks[j](
                    virtual_tokens, point_tokens, mask=mask
                )
                virtual_tokens = self.space_virtual_blocks[j](virtual_tokens)
                point_tokens = self.space_point2virtual_blocks[j](
                    point_tokens, virtual_tokens, mask=mask
                )
                space_tokens = torch.cat([point_tokens, virtual_tokens], dim=1)
                tokens = space_tokens.view(B, T, N, -1).permute(
                    0, 2, 1, 3
                )  # (B T) N C -> B N T C
                j += 1

        if self.add_space_attn:
            tokens = tokens[:, : N - self.num_virtual_tracks]

        tokens = tokens + init_tokens # 加了一个原始的，普通的cotracker没有 残差

        flow = self.flow_head(tokens)
        return flow


class CorrBlock:
    def __init__(
        self,
        fmaps,
        num_levels=4,
        radius=4,
        multiple_track_feats=False,
        padding_mode="zeros",
    ):
        B, S, C, H, W = fmaps.shape
        self.S, self.C, self.H, self.W = S, C, H, W
        self.padding_mode = padding_mode
        self.num_levels = num_levels
        self.radius = radius
        self.fmaps_pyramid = []
        self.multiple_track_feats = multiple_track_feats

        self.fmaps_pyramid.append(fmaps)
        for i in range(self.num_levels - 1):
            fmaps_ = fmaps.reshape(B * S, C, H, W)
            fmaps_ = F.avg_pool2d(fmaps_, 2, stride=2)
            _, _, H, W = fmaps_.shape
            fmaps = fmaps_.reshape(B, S, C, H, W)
            self.fmaps_pyramid.append(fmaps)

    def sample(self, coords):
        r = self.radius
        B, S, N, D = coords.shape
        assert D == 2

        H, W = self.H, self.W
        out_pyramid = []
        for i in range(self.num_levels):
            corrs = self.corrs_pyramid[i]  # B, S, N, H, W
            *_, H, W = corrs.shape

            dx = torch.linspace(-r, r, 2 * r + 1)
            dy = torch.linspace(-r, r, 2 * r + 1)
            delta = torch.stack(
                torch.meshgrid(dy, dx, indexing="ij"), axis=-1
            ).to(coords.device)

            centroid_lvl = coords.reshape(B * S * N, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corrs = bilinear_sampler(
                corrs.reshape(B * S * N, 1, H, W),
                coords_lvl,
                padding_mode=self.padding_mode,
            )
            corrs = corrs.view(B, S, N, -1)

            out_pyramid.append(corrs)

        out = torch.cat(out_pyramid, dim=-1).contiguous()  # B, S, N, LRR*2 (B, N, S, 层数*（LRR**2）=（4）*9**2)
        return out

    def corr(self, targets):
        B, S, N, C = targets.shape
        if self.multiple_track_feats:
            targets_split = targets.split(C // self.num_levels, dim=-1)
            B, S, N, C = targets_split[0].shape

        assert C == self.C
        assert S == self.S

        fmap1 = targets

        self.corrs_pyramid = []
        for i, fmaps in enumerate(self.fmaps_pyramid):
            *_, H, W = fmaps.shape
            fmap2s = fmaps.view(B, S, C, H * W)  # B S C H W ->  B S C (H W)
            if self.multiple_track_feats:
                fmap1 = targets_split[i]
            corrs = torch.matmul(fmap1, fmap2s)
            corrs = corrs.view(B, S, N, H, W)  # B S N (H W) -> B S N H W
            corrs = corrs / torch.sqrt(torch.tensor(C).float())
            self.corrs_pyramid.append(corrs)


class EfficientCorrBlock:
    def __init__(self, fmaps, num_levels=4, radius=4):
        B, S, C, H, W = fmaps.shape
        self.num_levels = num_levels
        self.radius = radius
        self.fmaps_pyramid = []
        self.fmaps_pyramid.append(fmaps)
        for i in range(self.num_levels - 1):
            fmaps_ = fmaps.reshape(B * S, C, H, W)
            fmaps_ = F.avg_pool2d(fmaps_, 2, stride=2)
            _, _, H, W = fmaps_.shape
            fmaps = fmaps_.reshape(B, S, C, H, W)
            self.fmaps_pyramid.append(fmaps)

    def sample(self, coords, target): # 查询点坐标、轨迹特征
        r = self.radius
        device = coords.device
        B, S, N, D = coords.shape
        assert D == 2
        target = target.permute(0, 1, 3, 2).unsqueeze(-1) # BSNC-》BSCN1

        out_pyramid = []

        for i in range(self.num_levels): #  针对金字塔的每一层进行操作
            pyramid = self.fmaps_pyramid[i]
            C, H, W = pyramid.shape[2:]
            centroid_lvl = (
                torch.cat(
                    [torch.zeros_like(coords[..., :1], device=device), coords],
                    dim=-1, # 在原始二维坐标前面拼接一列全零（对应于“深度”或额外维度），使得每个坐标扩展为三维向量。
                ).reshape(B * S, N, 1, 1, 3) # B S N 3->B*S N 1 1 3
                / 2**i # 除以 2𝑖2i  来匹配当前金字塔层因下采样带来的坐标缩放
            )

            dx = torch.linspace(-r, r, 2 * r + 1, device=device)
            dy = torch.linspace(-r, r, 2 * r + 1, device=device)
            xgrid, ygrid = torch.meshgrid(dy, dx, indexing="ij")
            zgrid = torch.zeros_like(xgrid, device=device)
            delta = torch.stack([zgrid, xgrid, ygrid], axis=-1)
            delta_lvl = delta.view(1, 1, 2 * r + 1, 2 * r + 1, 3)
            coords_lvl = centroid_lvl + delta_lvl #B*S N 2*r+1 2*r+1 3 将偏移量 delta_lvl 与中心坐标 centroid_lvl 相加，得到每个查询点在当前金字塔层内邻域的所有采样坐标。
            pyramid_sample = bilinear_sampler(
                pyramid.reshape(B * S, C, 1, H, W), coords_lvl
            ) # 先采样，标准的corr是先点乘 后采样 -> B*S N 2*r+1 2*r+1 3在 B*S, C, 1, H, W 采样 得到 B*S N 2*r+1 2*r+1 C

            corr = torch.sum(
                target * pyramid_sample.reshape(B, S, C, N, -1), dim=2
            ) # BSCN1 * BSCN(2*r+1)**2 将采样得到的特征（reshape 后形状为 B x S x C x N x (2r+1)²）与 target 特征做逐元素乘法，再在通道维度（C）上求和 B x S x N x (2r+1)²）
            corr = corr / torch.sqrt(torch.tensor(C).float())
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)  # B, S, N, LRR*2*num_levels
        return out


# # Inside the tracker forward funciton:
# if self.efficient_corr:
#     corr_block = EfficientCorrBlock(
#         fmaps,
#         num_levels=4,
#         radius=3,
#         padding_mode="border",
#     )
# else:
#     corr_block = CorrBlock(
#         fmaps,
#         num_levels=4,
#         radius=3,
#         padding_mode="border",
#     )
# if self.efficient_corr:
#     fcorrs = corr_block.sample(coords, track_feat)
# else:
#     corr_block.corr(track_feat)
#     fcorrs = corr_block.sample(coords)
