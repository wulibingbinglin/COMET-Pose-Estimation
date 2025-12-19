"""这个版本 就是融合pizza 的版本
特征提取 不融合
对轨迹和rgb进行cross attention得到融合编码
然后 特征+时间的融合编码+轨迹 =rgb输入 经过trunk后得到更深层的rgb

主要相对于8是用3个头去预测
相对于9 就是 服务器上的10错了 没有trunk 但是有+轨迹  但我的10全加上 也打开trunk  现在全打开
"""

import logging
from collections import defaultdict
from dataclasses import field, dataclass

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from einops import rearrange, repeat
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

from minipytorch3d.rotation_conversions import quaternion_to_matrix
from .modules import AttnBlock, CrossAttnBlock, Mlp, ResidualBlock

from .utils import (
    get_2d_sincos_pos_embed,
    PoseEmbedding,
    pose_encoding_to_camera,
    camera_to_pose_encoding, get_1d_sincos_pos_embed, camera_to_pose_encoding2, pose_encoding_to_camera2,
)

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class FeatureFusion(nn.Module):
    def __init__(self, rgb_dim, fmap_dim, fusion_type='adaptive'):
        super().__init__()
        self.fusion_type = fusion_type
        # 特征投影层
        self.fmap_proj = nn.Sequential(
            nn.Linear(fmap_dim, rgb_dim),
            nn.LayerNorm(rgb_dim),  # 添加归一化提高稳定性
            nn.ReLU()
        )
        self.fusion_layer = nn.Linear(rgb_dim * 2, rgb_dim)
        self.out_norm = nn.LayerNorm(rgb_dim)  # 融合后再加一层归一化

    def forward(self, rgb_feat, fmaps):
        # 1. 改进特征池化
        B, S, C_f, H, W = fmaps.shape
        # 使用多尺度池化捕获不同尺度的信息
        fmaps_global = F.adaptive_avg_pool2d(
            fmaps.reshape(B * S, C_f, H, W),
            output_size=(1, 1)
        )
        fmaps_local = F.adaptive_max_pool2d(  # 添加最大池化捕获显著特征
            fmaps.reshape(B * S, C_f, H, W),
            output_size=(1, 1)
        )
        fmaps_pooled = (fmaps_global + fmaps_local).reshape(B, S, C_f)

        # 2. 特征转换
        fmaps_transformed = self.fmap_proj(fmaps_pooled)

        out = self.fusion_layer(torch.cat([rgb_feat, fmaps_transformed], dim=-1))
        out = self.out_norm(out)
        return out


class TrajectoryEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, traj):  # traj: [B, S, N, in_dim]
        return self.mlp(traj)  # [B, S, N, out_dim]


class CameraPredictor(nn.Module):
    def __init__(
            self,
            hidden_size=768,  # Transformer 隐藏层大小（特征维度） 默认值
            num_heads=8,  # Transformer 多头注意力的头数
            mlp_ratio=4,  # MLP 层的扩展比例
            z_dim: int = 768,  # 就是经过dino得到特征向量的维度 默认就是768
            down_size=336,  # 图像降采样尺寸 输入图像降采样的目标尺寸 符合dino的尺度，且计算量变小
            att_depth=4,  # Transformer 自注意力层和交叉注意力 的深度
            trunk_depth=4,  # 干（trunk）部分的 Transformer 层数
            backbone="dinov2b",  # 选择骨干网络（如 DINO）
            pose_encoding_type="absT_quaR_OneFL",  # "absT_quaR_OneFL",  # 相机位姿编码方式
            cfg=None,  # 额外的配置 cfg
    ):
        super().__init__()
        self.cfg = cfg

        self.att_depth = att_depth
        self.down_size = down_size
        self.pose_encoding_type = pose_encoding_type  # 'absT_quaR_OneFL'

        if self.pose_encoding_type == "absT_quaR":  # 7维
            # 只包含外参：绝对平移(3) + 四元数旋转(4)
            self.target_dim = 7
        if self.pose_encoding_type == "absT_quaR_OneFL":  # absT（绝对平移）+ quaR（四元数旋转）+ OneFL（某种特定的额外参数）
            self.target_dim = 8
        if self.pose_encoding_type == "absT_quaR_logFL":  # logFL 可能是某种对焦距（focal length）的对数编码。
            self.target_dim = 9

        self.backbone = self.get_backbone(backbone)  # get_backbone(backbone) 负责加载骨干网络，例如 DINO（DINOv2b）骨干网络用于提取图像的深层特征

        for param in self.backbone.parameters():
            param.requires_grad = False
            # 这样模型不会更新 backbone 的参数，而是使用预训练的特征。
            # 只训练姿态预测相关的部分，减少计算成本，提高稳定性。

        self.input_transform = Mlp(  # mlp
            in_features=z_dim, out_features=hidden_size, drop=0
        )
        self.norm1 = nn.LayerNorm(  # 归一化
            hidden_size, elementwise_affine=False, eps=1e-6
        )

        self.norm2 = nn.LayerNorm(  # 归一化
            hidden_size, elementwise_affine=False, eps=1e-6
        )

        # sine and cosine embed for camera parameters
        self.embed_pose = PoseEmbedding(
            target_dim=self.target_dim,  # 7
            # n_harmonic_functions=(hidden_size // self.target_dim) // 2, # 48 这就是下面的式子反求出来的,就是需要最终的位置编码维度和hidden_size保持一致
            # 这里有bug 明天改，反正需要一致
            n_harmonic_functions=int(hidden_size / self.target_dim / 2),
            append_input=False,  # 这是false所以不加上原先的input(1,8,8)
        )  # 所以输出是(1,8,(2*n_harmonic_functions*dim + append_input*dim))=(1,8,768)

        self.pose_token = nn.Parameter(
            torch.zeros(1, 1, 1, hidden_size)
        )  # register 可学习的参数（Token），用于 Transformer 处理姿态信息

        self.pose_branch = Mlp(
            in_features=hidden_size,  # 输入
            hidden_features=hidden_size * 2,
            out_features=4,  # 输出
            drop=0,
        )

        self.ffeat_updater = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size)
        )

        self.self_att = nn.ModuleList(
            [  # 多个 AttnBlock 组成的 Transformer 自注意力层。
                AttnBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_class=nn.MultiheadAttention,
                )
                for _ in range(self.att_depth)
            ]
        )

        self.pose_branch_scale = nn.Parameter(torch.ones(1) * 0.1)

        self.cross_att = nn.ModuleList(
            [
                CrossAttnBlock(
                    hidden_size, hidden_size, num_heads, mlp_ratio=mlp_ratio
                )
                for _ in range(self.att_depth)
            ]
        )
        self.cross_attn_block = nn.ModuleList(
            [
                CrossAttnBlock(
                    hidden_size, hidden_size, num_heads, mlp_ratio=mlp_ratio
                )
                for _ in range(self.att_depth)
            ]
        )

        self.trunk = nn.Sequential(
            *[
                AttnBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_class=nn.MultiheadAttention,
                )
                for _ in range(trunk_depth)
            ]
        )

        self.gamma = 0.8
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 初始值为0.5，范围 [0,1]

        nn.init.normal_(self.pose_token, std=1e-6)
        # 通过 PyTorch 的初始化函数 normal_ 初始化 pose_token 这通常用于初始化权重，确保它们从接近零的地方开始训练，有助于避免训练初期的梯度问题。

        for name, value in (  # _RESNET_MEAN 和 _RESNET_STD: 这两个值可能是存储在配置文件中的预先定义的 ResNet 图像标准化的均值和标准差。
                ("_resnet_mean", _RESNET_MEAN),
                ("_resnet_std", _RESNET_STD),
        ):
            self.register_buffer(
                name,
                torch.FloatTensor(value).view(1, 3, 1, 1),
                persistent=False,
            )
            # register_buffer: 这是一个 PyTorch 的方法，它将 name 作为缓冲区的名字注册，并将 value（此处是 ResNet 的均值和标准差）作为其值。
            # 注册的缓冲区将不会参与梯度计算，因此不会更新，也不会存储在优化器的状态中。
            # persistent=False 表示该缓冲区不会被保存到模型的状态字典中（即在保存模型时不保存该缓冲区）。
            # register_buffer 主要用于存储 不需要梯度更新的变量，但又希望它们 随模型一起移动到 GPU/CPU，
            # 并在 state_dict（模型参数字典）中可选地存储或跳过。设置 persistent=False 使得这些变量 不会被保存到模型的 state_dict 中，但仍然可以在推理时使用。
        if not hasattr(self, 'feature_fusion'):
            # 延迟初始化融合模块
            self.feature_fusion = FeatureFusion(
                rgb_dim=768,  # 根据实际维度调整
                fmap_dim=128,  # 根据实际维度调整
                fusion_type='adaptive'  # 可选: 'simple', 'adaptive', 'linear'
            )

        # 轨迹特征转换器
        # self.trajectory_encoder = nn.Sequential(
        #     nn.Linear(2, 32),  # 2D轨迹点坐标
        #     nn.LayerNorm(32),
        #     nn.ReLU(),
        #     nn.Linear(32, 64)
        # )

        # 轨迹置信度注意力层
        self.confidence_attention = nn.Sequential(
            nn.Linear(1, 32),  # 轨迹置信度
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        # 在 __init__ 的超参数配置部分加入：
        self.motion_weight = self.cfg.get("motion_weight", 0.1)
        # 相机运动编码器 - 将相机运动编码到与轨迹运动相同的特征空间
        self.camera_motion_encoder = nn.Sequential(
            # 输入维度是8: 3(平移差异) + 4(相对旋转)
            nn.Linear(7, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 2)  # 输出维度为2,与轨迹motion_consistency保持一致
        )
        self.motion_encoder = nn.Sequential(
            nn.Linear(3, 32),  # 输入轨迹每一帧的运动（x, y），所以是2维
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 2)  # 输出与 motion_consistency 的特征维度保持一致（最终是2维）
        )

        self.traj_encoder = TrajectoryEncoder(2, 256, 768)
        self.track_context_proj = nn.Sequential(
            nn.Linear(128, 768),
            nn.GELU(),
            nn.LayerNorm(768)
        )

        self.traj_encoder_norm = nn.LayerNorm(128)
        self.traj_context_norm = nn.LayerNorm(768)

        self.pose_embed_norm = nn.LayerNorm(768)
        self.pose_embed_scale = nn.Parameter(torch.ones(1) * 0.05)  # 初始更小

        self.fc_translation2d = nn.Linear(768, 2)
        self.fc_depth = nn.Linear(768, 1)

        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.xavier_uniform_(m.weight)
        #         if m.bias is not None:
        #             nn.init.zeros_(m.bias)

    def forward(
                self,
                reshaped_image, #（BN）3HW
                preliminary_cameras=None, # 预估的相机参数（如果提供的话）
                iters=4,
                batch_size=None,# 1
                rgb_feat_init=None,
                gt_cameras=None,
                fmaps=None,
                pred_trajectories=None,
                track_confidence=None,
                debug = False):
        """
        # reshaped_image: Bx3xHxW. The values of reshaped_image are within [0, 1]
        # preliminary_cameras: cameras in opencv coordinate.
        """

        torch.autograd.set_detect_anomaly(True)
        if rgb_feat_init is None:
            rgb_feat, B, S, C = self.get_2D_image_features(
                reshaped_image, batch_size  # 用 vit方法初始化的 feature (1 8 768)
            )
        else:
            rgb_feat = rgb_feat_init  #
            B, S, C = rgb_feat.shape  # 1 8 768

        # —— MONITOR：记录 get_2D_image_features 输出 ——
        if debug:
            print("1. after get_2D_image_features, rgb_feat:",
                  rgb_feat.min().item(), rgb_feat.max().item(), rgb_feat.mean().item())
            if (rgb_feat.requires_grad):
                rgb_feat.register_hook(lambda g: print("[GRAD HOOK] rgb_feat(after feature) grad norm:", g.norm().item()))


        # —— MONITOR：记录加上 time_emb 之后 ——
        if debug:
            print("4. after time_emb, rgb_feat:",
                  rgb_feat.min().item(), rgb_feat.max().item(), rgb_feat.mean().item())
            if rgb_feat.requires_grad:
                rgb_feat.register_hook(lambda g: print("[GRAD HOOK] rgb_feat(after time) grad norm:", g.norm().item()))

        if pred_trajectories is not None:
            traj_encoded = self.traj_encoder(pred_trajectories)  # [B, S, N, 128]
            B, S, N, D = traj_encoded.shape
            confidence_weights = self.confidence_attention(track_confidence.unsqueeze(-1))  # [B, S, N, 1]
            traj_context = (traj_encoded * confidence_weights)
            if debug:
                print("6. traj_context:", traj_context.min().item(), traj_context.max().item(), traj_context.mean().item())
                if traj_context.requires_grad:
                    traj_context.register_hook(lambda g: print("[GRAD HOOK] 加权之后的轨迹特征:", g.norm().item()))
            traj_flat = traj_context.reshape(B * S, N, D)
            rgb_flat = rgb_feat.reshape(B * S, 1, D)
            for block in self.cross_attn_block:
                rgb_flat = block(rgb_flat, traj_flat)  # 逐层更新
            rgb_flat = rgb_flat.reshape(B, S, D)
            if debug:
                print("7. fused_feat:", rgb_flat.min().item(), rgb_flat.max().item(), traj_context.mean().item())
                if rgb_flat.requires_grad:
                    rgb_flat.register_hook(
                        lambda g: print("[GRAD HOOK] cross attention 轨迹和feature的结果:", g.norm().item()))
            rgb_feat = rgb_feat + rgb_flat  # [B, S, rgb_dim]
            if debug:
                # —— MONITOR：记录叠加轨迹后 ——
                print("7. after adding traj_context, rgb_feat:",
                      rgb_feat.min().item(), rgb_feat.max().item(), rgb_feat.mean().item())
                if rgb_feat.requires_grad:
                    rgb_feat.register_hook(lambda g: print("[GRAD HOOK]attention和rgb残差之后的结果:", g.norm().item()))


        if gt_cameras is not None:
            gt_pose_enc = camera_to_pose_encoding2(gt_cameras, pose_encoding_type=self.pose_encoding_type)
            # B*S 8

        loss_pose = 0
        trans_loss = 0
        rot_loss = 0

        time_emb = get_1d_sincos_pos_embed(C, S)  # (1, S, C)
        time_emb = time_emb.expand(B, -1, -1).to(rgb_feat.device)  # 转为 (B, S, C)
        if debug:
            print(f"time_emb: {time_emb.min().item(), time_emb.max().item()}")
            if(time_emb.requires_grad):
                time_emb.register_hook(lambda g:print(f"[GRAD HOOK] time_emb:", g.norm().item()))
        rgb_feat = rgb_feat + time_emb

        if debug:
            print(f"11_itegb_feat(after pose_embed):",
                  rgb_feat.min().item(), rgb_feat.max().item(), rgb_feat.mean().item())
            if rgb_feat.requires_grad:
                rgb_feat.register_hook(
                lambda g: print(f"[GRAD HOOK] rgb_feat(after pose_embed)_grad_norm:",
                                            g.norm().item()))

        # # Run trunk transformers on rgb_feat
        rgb_feat = self.trunk(rgb_feat)  # muti headattention 提取更加抽象的特征表示 BS（hidden_size ）

        # Predict the delta feat and pose encoding at each iteration
        pred_rotation = self.pose_branch(rgb_feat) # mlp用来根据处理后的图像特征（rgb_feat）预测相机位姿的变化量（delta）BS（hidden_size + self.target_dim）

        if debug:
            print(f"13delta:",
                  pred_rotation.min().item(),pred_rotation.max().item(), pred_rotation.mean().item())
            if pred_rotation.requires_grad:
                pred_rotation.register_hook(lambda g: print(f"[GRAD HOOK] delta_grad_iter norm:", g.norm().item()))
        pred_uv = self.fc_translation2d(rgb_feat)
        if debug:
            print(f"14uv:",
                  pred_uv.min().item(), pred_uv.max().item(), pred_uv.mean().item())
            if pred_uv.requires_grad:
                pred_uv.register_hook(lambda g: print(f"[GRAD HOOK] uv_grad_norm:", g.norm().item()))
        pred_d = self.fc_depth(rgb_feat)
        if debug:
            print(f"15depth:",
                  pred_d.min().item(), pred_d.max().item(), pred_d.mean().item())
            if pred_d.requires_grad:
                pred_d.register_hook(lambda  g: print(f"[GRAD HOOK] depth_grad_norm:", g.norm().item()))
        pred_uvd = torch.cat([pred_uv, pred_d], dim=-1)
        if torch.isnan(pred_uvd).any() or torch.isinf(pred_uvd).any():
            print("Warning: NaN or Inf detected in pred_uvd")

        pred_rotation = F.normalize(
            pred_rotation.clone(),  # 任意形状的四元数子张量
            p=2,  # L2 范数
            dim=-1,  # 在最后一维（4 维）上归一化
            eps=1e-8  # 防止除以零
        )

        def check_tensor(tensor, name):
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                print(f"Warning: {name} contains NaN or Inf")
                print(f"{name} stats: min={tensor.min()}, max={tensor.max()}, mean={tensor.mean()}")

        if gt_cameras:
            # 根据航天器姿态任务的领域知识，设定不同的损失权重
            # 通常旋转（例如四元数）误差比平移误差更敏感，因此赋予更高的权重
            weight_trans = self.cfg.get("weight_trans", 1)  # 平移损失权重，默认为 1.0
            weight_rot = self.cfg.get("weight_rot", 2)  # 旋转损失权重，默认为 2.0
            pred_trans = pred_uvd[:, 1:, :].clone().reshape(-1, 3)  # [S-1, 3]
            gt_trans = gt_pose_enc[1:, :3].clone()  # [ S-1, 3]
            pred_rot = pred_rotation[:, 1:, :].clone().reshape(-1, 4)  # [S-1, 4]
            gt_rot = gt_pose_enc[1:, 3:7].clone()  # [ S-1, 4]]
            trans_loss = F.mse_loss(
                pred_trans,
                gt_trans
            )* (10 ** 2)
            rot_loss = F.mse_loss(
                pred_rot,
                gt_rot
            )* (10 ** 2)
            # 检查预测值
            loss_pose = (weight_trans * trans_loss + weight_rot * rot_loss)
            # check_tensor(pred_trans, "pred_trans")
            # check_tensor(gt_trans, "gt_trans")
            # check_tensor(pred_rot, "pred_rot")
            # check_tensor(gt_rot, "gt_rot")
            #
            # # 检查损失计算过程
            # raw_trans_loss = F.mse_loss(pred_trans, gt_trans)
            # raw_rot_loss = F.mse_loss(pred_rot, gt_rot)
            # print(f"Raw losses - trans: {raw_trans_loss.item()}, rot: {raw_rot_loss.item()}")
            #
            # trans_loss = raw_trans_loss * (10 ** 2)
            # rot_loss = raw_rot_loss * (10 ** 2)
            # print(f"Scaled losses - trans: {trans_loss.item()}, rot: {rot_loss.item()}")
            #
            # loss_pose = (weight_trans * trans_loss + weight_rot * rot_loss)
            # print(f"Final loss_pose: {loss_pose.item()}")

        # 确保第一帧始终保持为零变换
        pred_uvd[:, 0, :] = 0  # 平移为零
        pred_rotation[:, 0, :] = torch.tensor([1, 0, 0, 0], device=pred_rotation.device)  # 单位四元数

        pred_pose_enc = torch.cat([pred_uvd,pred_rotation], dim=-1)

        # 计算平均姿态损失。将累积的姿态损失 loss_pose 除以迭代次数 iters，
        # 然后乘以一个系数 10。这可能是为了将损失值缩放到一个合适的范围。

        # print("类型",self.cfg.train.dataset)
        pred_cameras = pose_encoding_to_camera2(
            pred_pose_enc,  # (1 8 8) 此时是相对姿态 四元数格式的
            gt_cameras=gt_cameras,
            pose_encoding_type=self.pose_encoding_type,  # 'absT_quaR_OneFL'
            to_OpenCV=False,
            intri_type=self.cfg.train.dataset
        )  # 将位姿编码转换为相机参数。具体来说，这可能包括相机的旋转矩阵、平移向量等，具体取决于如何定义位姿和相机模型
        # 获取参考帧（第一帧）的四元数和平移

        pose_predictions = {
            "pred_pose_enc": pred_pose_enc.reshape(-1, 7),  # 这个是相对姿态的8元式
            "gt_pose_enc": gt_pose_enc,  # 相对真值的8元式
            "pred_cameras": pred_cameras,  # 绝对姿态的8元式
            "loss": loss_pose,
            "loss_trans": trans_loss,
            "loss_rot": rot_loss
        }

        return pose_predictions



    def quaternion_inverse(self, quaternion):
        """
        计算四元数的逆
        q^(-1) = conjugate(q) / norm(q)^2
        对于单位四元数，norm(q) = 1，所以直接取共轭即可
        """
        # 假设输入是单位四元数 [w, x, y, z]
        inv_quaternion = quaternion.clone()
        # 对虚部取负
        inv_quaternion[..., 1:] = -inv_quaternion[..., 1:]
        return inv_quaternion

    def quaternion_multiply(self, q1, q2):
        """
        四元数乘法
        q1 * q2 = [
            w1w2 - x1x2 - y1y2 - z1z2,
            w1x2 + x1w2 + y1z2 - z1y2,
            w1y2 - x1z2 + y1w2 + z1x2,
            w1z2 + x1y2 - y1x2 + z1w2
        ]
        """
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

        result = torch.zeros_like(q1)
        result[..., 0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2  # w
        result[..., 1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2  # x
        result[..., 2] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2  # y
        result[..., 3] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2  # z

        return result

    def get_identity_camera(self, device):
        """
        返回表示单位变换的相机参数
        """
        identity_camera = torch.zeros(self.target_dim, device=device)
        # 设置单位四元数 [1, 0, 0, 0]
        identity_camera[3:7] = torch.tensor([1, 0, 0, 0], device=device)
        return identity_camera

    # def compute_camera_motion(self, pose_enc):
    #     """
    #     计算相机运动
    #     Args:
    #         pose_enc: [B, S, target_dim] 相机姿态编码
    #     Returns:
    #         camera_motion: [B, S-1, 2] 相机运动特征
    #     """
    #     # 计算相邻帧之间的姿态差异
    #     motion = pose_enc[:, 1:] - pose_enc[:, 0:1]  # [B, S-1, target_dim]
    #
    #     # 将姿态差异映射到与轨迹运动相同的特征空间
    #     camera_motion = self.camera_motion_encoder(motion)  # [B, S-1, 2]
    #
    #     return camera_motion


    def process_trajectory_features(self, pred_trajectories):
        """处理轨迹预测结果，提取运动特征"""
        # pred_trajectories: [B, S, N, 2] - N个特征点的2D轨迹
        # track_confidence: [B, S, N] - 每个轨迹点的置信度

        # 1. 计算相对运动 指的是在连续两帧之间，关键点坐标的变化
        relative_motion = pred_trajectories[:, 1:] - pred_trajectories[:, 0:1]  # [B, S-1, N, 2]

        # 2. 计算运动一致性 在这里，计算沿着 N 维（即每一帧中所有关键点）的标准差，可以反映出各关键点运动是否一致：
        # 如果所有关键点在一帧内运动比较一致，则标准差较小；
        # 如果关键点运动分散，标准差较大，则说明运动不一致。 每个元素表示在该帧间（相邻帧）的 x 和 y 方向上的运动一致性。
        motion_consistency = torch.std(relative_motion, dim=2)  # [B, S-1, 2]

        return motion_consistency  # [B, S-1, 2] # [B, S, N, 1]

    def compute_trajectory_motion(self, trajectories, eps=1e-6):
        # 绝对disp = ti - t0
        disp = trajectories[:, 1:] - trajectories[:, :1]  # [B,S-1,N,2]
        mag = disp.norm(dim=-1, keepdim=True)  # [B,S-1,N,1]
        disp_dir = disp / (mag + eps)  # [B,S-1,N,2]
        mean_dir = disp_dir.mean(dim=2)  # [B,S-1,2]
        mean_mag = mag.mean(dim=2)  # [B,S-1,1]
        feats = torch.cat([mean_dir, mean_mag.log()], dim=-1)  # [B,S-1,3]
        return self.motion_encoder(feats)  # 输入 Linear(3→...)

    def quaternion_loss(self, pred_q, gt_q):
        """计算四元数损失，考虑四元数的双覆盖性质"""
        # 确保四元数归一化 这相当于rot_loss = F.smooth_l1_loss(rot_pred.reshape(-1, 4), rot_gt.reshape(-1, 4))
        # 的替代，至于要不要替代，还有想法
        pred_q = F.normalize(pred_q, dim=-1)
        gt_q = F.normalize(gt_q, dim=-1)

        # 考虑q和-q表示相同旋转
        loss_pos = torch.mean((pred_q - gt_q) ** 2)
        loss_neg = torch.mean((pred_q + gt_q) ** 2)

        return torch.min(loss_pos, loss_neg)

    # def compute_trajectory_motion(self, trajectories):
    #     """从轨迹预测计算运动模式"""
    #     # 计算连续帧之间的运动
    #     motion = trajectories[:, 1:] - trajectories[:, 0:1]
    #     # 如果你需要提取主要运动方向和幅度，可以将其压缩成一个更紧凑的表示，例如：
    #     # motion = F.normalize(motion, dim=-1).mean(dim=-2)   # 只保留方向
    #     # 通过求平均的方式获得主要运动的方向和幅度
    #     return self.motion_encoder(motion)

    def adjust_pose_update(self, delta_pose, trajectory_motion, camera_motion):
        """根据轨迹运动和相机运动的一致性调整姿态更新"""
        consistency = F.cosine_similarity(trajectory_motion, camera_motion, dim=-1)
        consistency = F.pad(consistency, (1, 0), mode="constant", value=1.0)
        # 对 delta_pose 每一帧进行一致性加权
        return delta_pose * consistency.unsqueeze(-1)

    def get_backbone(self, backbone):
        """
        Load the backbone model.返回类的实例
        """
        if backbone == "dinov2s":  # DINOv2小版本，基于ViT-small
            return torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vits14_reg"
            )
        elif backbone == "dinov2b":  # DINOv2大版本，基于ViT-base
            return torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vitb14_reg"
            )  # 这是 PyTorch 的一个函数，用于从指定的 GitHub 仓库下载和加载模型。facebookresearch/dinov2 是 DINOv2 模型的实现库
        # ckpt_path = "checkpoints/dinov2_vitb14_reg4_pretrain.pth"
        #             assert os.path.exists(ckpt_path), f"Checkpoint not found at {ckpt_path}"
        #             model = vit_base(patch_size=14)
        else:
            raise NotImplementedError(f"Backbone '{backbone}' not implemented")

    def _resnet_normalize_image(self, img: torch.Tensor) -> torch.Tensor:
        return (img - self._resnet_mean) / self._resnet_std

    def get_2D_image_features(self, reshaped_image, batch_size):
        # Get the 2D image features 处理后的图像特征，包含空间和时间上的嵌入信息，能够用于后续的任务
        if reshaped_image.shape[-1] != self.down_size:
            reshaped_image = F.interpolate(  # 检查输入图像的大小是否与预期的下采样尺寸（self.down_size）一致
                reshaped_image,
                (self.down_size, self.down_size),
                mode="bilinear",
                align_corners=True,
            )

        with torch.no_grad():
            reshaped_image = self._resnet_normalize_image(reshaped_image)  # 通过 self._resnet_normalize_image 函数对图像进行标准化
            rgb_feat = self.backbone(reshaped_image, is_training=True)
            # B x P x C
            rgb_feat = rgb_feat["x_norm_patchtokens"]  # 8，576，768

        rgb_feat = self.input_transform(rgb_feat)  # 8，576，768
        rgb_feat = self.norm2(rgb_feat)  # 8，576，768

        rgb_feat = rearrange(rgb_feat, "(b s) p c -> b s p c", b=batch_size)  # 8，576，768
        # (B, S, P, C)，即批次大小 B、帧数 S、每帧被切分成 P 个 Patch、每个 Patch 的通道数

        B, S, P, C = rgb_feat.shape  # （1，8，576，768）
        patch_num = int(math.sqrt(P))  # 二维网格的边长 P=patch_num*patch_num=24*24=576

        # add embedding of 2D spaces
        pos_embed = get_2d_sincos_pos_embed(  # 为图像特征添加 2D 的正余弦位置嵌入
            C, grid_size=(patch_num, patch_num)  # C=768, grid_size定义网格大小（即图像被划分为多少个 patch）
        ).permute(0, 2, 3, 1)[None]  # （1 768 24 24）-》 （1，1 24 24 768）
        pos_embed = pos_embed.reshape(1, 1, patch_num * patch_num, C).to(
            rgb_feat.device
        )  # （1 1 576 768）

        rgb_feat = rgb_feat + pos_embed  # # (B, S, P, C')  （1 8 576 768）不是连接 是只加上去 所以大小没变

        # register for pose
        pose_token = self.pose_token.expand(B, S, -1, -1)  # (1, 8, 1,768)->(1，8, 1, 768)
        rgb_feat = torch.cat([pose_token, rgb_feat], dim=-2)  # 1, 8, 577，768

        B, S, P, C = rgb_feat.shape

        for idx in range(self.att_depth):  # (1,8,577,768)->(1,8,577,768)-
            # self attention
            # 在 rearrange 里，只有涉及变形（reshaping）或者新创建的维度才需要显式指定，而 p 和 c 在输入和输出中没有变化，所以不需要额外定义
            rgb_feat = rearrange(rgb_feat, "b s p c -> (b s) p c", b=B, s=S)
            rgb_feat = self.self_att[idx](rgb_feat)
            rgb_feat = rearrange(rgb_feat, "(b s) p c -> b s p c", b=B, s=S)

            feat_0 = rgb_feat[:, 0]
            feat_others = rgb_feat[:, 1:]

            # cross attention 将第一个帧的特征（feat_0）作为查询，与其他帧的特征（feat_others）进行交互，从而强化帧间的依赖关系
            feat_others = rearrange(
                feat_others, "b m p c -> b (m p) c", m=S - 1, p=P
            )
            feat_others = self.cross_att[idx](feat_others, feat_0)

            feat_others = rearrange(
                feat_others, "b (m p) c -> b m p c", m=S - 1, p=P
            )
            rgb_feat = torch.cat([rgb_feat[:, 0:1], feat_others], dim=1)  # 最终将跨注意力后的特征与第一个帧的特征拼接在一起。
            # BSPC

        rgb_feat = rgb_feat[:, :, 0]  # (1,8,1,768)

        return rgb_feat, B, S, C





