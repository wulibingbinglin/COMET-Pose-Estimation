# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Modified from https://github.com/facebookresearch/PoseDiffusion
# and https://github.com/facebookresearch/co-tracker/tree/main


import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Union
from einops import rearrange, repeat


from minipytorch3d.harmonic_embedding import HarmonicEmbedding

from minipytorch3d.cameras import CamerasBase, PerspectiveCameras
from minipytorch3d.rotation_conversions import (
    matrix_to_quaternion,
    quaternion_to_matrix, quaternion_multiply, quaternion_invert, quaternion_apply,
)
from train_eval_func import QuaternionCameras

# from pytorch3d.renderer import HarmonicEmbedding
# from pytorch3d.renderer.cameras import CamerasBase, PerspectiveCameras
# from pytorch3d.transforms.rotation_conversions import matrix_to_quaternion, quaternion_to_matrix



EPS = 1e-9


def get_1d_sincos_pos_embed_from_grid(
    embed_dim: int, pos: torch.Tensor
) -> torch.Tensor:
    """
    This function generates a 1D positional embedding from a given grid using sine and cosine functions.
    从给定的位置生成一个一维的正弦余弦位置嵌入
    Args:
    - embed_dim: The embedding dimension. 应该是一个偶数，以便将其分成两个部分来分别处理正弦和余弦
    - pos: The position to generate the embedding from.

    Returns:
    - emb: The generated 1D positional embedding.
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.double)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb[None].float()


def get_2d_embedding(
    xy: torch.Tensor, C: int, cat_coords: bool = True
) -> torch.Tensor:
    """
    This function generates a 2D positional embedding from given coordinates using sine and cosine functions.

    Args:
    - xy: The coordinates to generate the embedding from.
    - C: The size of the embedding.
    - cat_coords: A flag to indicate whether to concatenate the original coordinates to the embedding.

    Returns:
    - pe: The generated 2D positional embedding.
    """
    B, N, D = xy.shape
    assert D == 2

    x = xy[:, :, 0:1]
    y = xy[:, :, 1:2]
    div_term = (
        torch.arange(0, C, 2, device=xy.device, dtype=torch.float32)
        * (1000.0 / C)
    ).reshape(1, 1, int(C / 2))

    pe_x = torch.zeros(B, N, C, device=xy.device, dtype=torch.float32)
    pe_y = torch.zeros(B, N, C, device=xy.device, dtype=torch.float32)

    pe_x[:, :, 0::2] = torch.sin(x * div_term)
    pe_x[:, :, 1::2] = torch.cos(x * div_term)

    pe_y[:, :, 0::2] = torch.sin(y * div_term)
    pe_y[:, :, 1::2] = torch.cos(y * div_term)

    pe = torch.cat([pe_x, pe_y], dim=2)  # (B, N, C*3)
    if cat_coords:
        pe = torch.cat([xy, pe], dim=2)  # (B, N, C*3+3)
    return pe

def create_intri_matrix(focal_length, principal_point):
    """
    Creates a intri matrix from focal length and principal point.

    Args:
        focal_length (torch.Tensor): A Bx2 or BxSx2 tensor containing the focal lengths (fx, fy) for each image.
        principal_point (torch.Tensor): A Bx2 or BxSx2 tensor containing the principal point coordinates (cx, cy) for each image.

    Returns:
        torch.Tensor: A Bx3x3 or BxSx3x3 tensor containing the camera matrix for each image.
    """

    if len(focal_length.shape) == 2:
        B = focal_length.shape[0]
        intri_matrix = torch.zeros(
            B, 3, 3, dtype=focal_length.dtype, device=focal_length.device
        )
        intri_matrix[:, 0, 0] = focal_length[:, 0]
        intri_matrix[:, 1, 1] = focal_length[:, 1]
        intri_matrix[:, 2, 2] = 1.0
        intri_matrix[:, 0, 2] = principal_point[:, 0]
        intri_matrix[:, 1, 2] = principal_point[:, 1]
    else:
        B, S = focal_length.shape[0], focal_length.shape[1]
        intri_matrix = torch.zeros(
            B, S, 3, 3, dtype=focal_length.dtype, device=focal_length.device
        )
        intri_matrix[:, :, 0, 0] = focal_length[:, :, 0]
        intri_matrix[:, :, 1, 1] = focal_length[:, :, 1]
        intri_matrix[:, :, 2, 2] = 1.0
        intri_matrix[:, :, 0, 2] = principal_point[:, :, 0]
        intri_matrix[:, :, 1, 2] = principal_point[:, :, 1]

    return intri_matrix

def closed_form_inverse_OpenCV(se3, R=None, T=None):
    """
    Computes the inverse of each 4x4 SE3 matrix in the batch.利用旋转矩阵特性快速求逆的方法

    Args:
    - se3 (Tensor): Nx4x4 tensor of SE3 matrices.

    Returns:
    - Tensor: Nx4x4 tensor of inverted SE3 matrices.


    | R t |
    | 0 1 |
    -->
    | R^T  -R^T t|
    | 0       1  |
    """
    if R is None:
        R = se3[:, :3, :3]

    if T is None:
        T = se3[:, :3, 3:]

    # Compute the transpose of the rotation
    R_transposed = R.transpose(1, 2)

    # -R^T t
    top_right = -R_transposed.bmm(T)

    inverted_matrix = torch.eye(4, 4)[None].repeat(len(se3), 1, 1)
    inverted_matrix = inverted_matrix.to(R.dtype).to(R.device)

    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right

    return inverted_matrix
def get_EFP(pred_cameras, image_size, B, S, default_focal=False):
    """
    Converting PyTorch3D cameras to extrinsics, intrinsics matrix

    Return extrinsics, intrinsics, focal_length, principal_point
    """
    scale = image_size.min()

    focal_length = pred_cameras.focal_length

    principal_point = torch.zeros_like(focal_length)

    focal_length = focal_length * scale / 2
    principal_point = (image_size[None] - principal_point * scale) / 2

    Rots = pred_cameras.R.clone()
    Trans = pred_cameras.T.clone()

    extrinsics = torch.cat([Rots, Trans[..., None]], dim=-1)

    # reshape
    extrinsics = extrinsics.reshape(B, S, 3, 4)
    focal_length = focal_length.reshape(B, S, 2)
    principal_point = principal_point.reshape(B, S, 2)

    # only one dof focal length
    if default_focal:
        focal_length[:] = scale
    else:
        focal_length = focal_length.mean(dim=-1, keepdim=True).expand(-1, -1, 2)
        focal_length = focal_length.clamp(0.2 * scale, 5 * scale)

    intrinsics = create_intri_matrix(focal_length, principal_point)

    return extrinsics, intrinsics


def pose_encoding_to_camera(
    pose_encoding, # 1 8 8
    pose_encoding_type="absT_quaR_OneFL", # 'absT_quaR_OneFL'
    log_focal_length_bias=1.8,
    min_focal_length=0.1,
    max_focal_length=30,
    return_dict=False,
    to_OpenCV=False,
    gt_cameras=None):
    """
    Args:
        pose_encoding: A tensor of shape `BxNxC`, containing a batch of
                        `BxN` `C`-dimensional pose encodings.
        pose_encoding_type: The type of pose encoding,
    """
    pose_encoding_reshaped = pose_encoding.reshape(
        -1, pose_encoding.shape[-1]
    )  # Reshape to BNxC (8 8)

    q_ref = gt_cameras.R[0]  # [4] 参考帧四元数
    T_ref = gt_cameras.T[0]  # [3] 参考帧平移
    q_ref = q_ref.unsqueeze(0).expand(pose_encoding_reshaped.shape[0], 4)  # [B, 4]
    T_ref = T_ref.unsqueeze(0).expand(pose_encoding_reshaped.shape[0], 3)  # [B, 3]

    if pose_encoding_type == "absT_quaR_OneFL":
        # 3 for absT, 4 for quaR, 1 for absFL
        # [absolute translation, quaternion rotation, normalized focal length]
        abs_T = pose_encoding_reshaped[:, :3]
        rel_q = pose_encoding_reshaped[:, 3:7]
        # 绝对四元数：q_abs = q_rel * q_ref
        abs_q = quaternion_multiply(rel_q, q_ref)  # [B, 4]
        # 绝对平移：T_abs = rotate(rel_T, q_ref) + T_ref
        abs_T = abs_T + T_ref  # [B, 3]

        # R = quaternion_to_matrix(R) # todo: 如果真值是旋转矩阵再将四元数转换为旋转矩阵 R
        focal_length = pose_encoding_reshaped[:, 7:8]
        focal_length = torch.clamp( # 确保焦距在 [min_focal_length, max_focal_length] 范围
            focal_length, min=min_focal_length, max=max_focal_length
        )
    else:
        raise ValueError(f"Unknown pose encoding {pose_encoding_type}")

    if to_OpenCV:
        ### From Pytorch3D coordinate to OpenCV coordinate:
        # I hate coordinate conversion
        R = abs_q.clone()
        abs_T = abs_T.clone() # 确保原始的旋转矩阵 R 和位移向量 abs_T 不会被修改。
        R[:, :, :2] *= -1
        abs_T[:, :2] *= -1 #理论上，最主要需要反转的是 Y 轴，因为 Pytorch3D 中 Y 轴指向上，而 OpenCV 中 Y 轴指向下。
        R = R.permute(0, 2, 1)

    if return_dict: # false
        return {"focal_length": focal_length, "R": R, "T": abs_T}

    pred_cameras = QuaternionCameras(
        focal_length=focal_length, R=abs_q, T=abs_T, device=abs_q.device
    ) # 这是一个类（通常用于处理相机模型），建立了一个 透视相机。PerspectiveCameras 是 PyTorch3D 库中的一个类，用来表示透视相机的内参和外参。
    return pred_cameras

def pose_encoding_to_camera3(
    pose_encoding,  # 形状: B x N x C
    pose_encoding_type=None,  # 编码类型
    log_focal_length_bias=1.8,
    min_focal_length=0.1,
    max_focal_length=30,
    return_dict=False,
    to_OpenCV=False,
    gt_cameras=None
):
    B, S, C = pose_encoding.shape
    # 强制将输入展平为 (B*S, C) 处理
    pose_encoding_reshaped = pose_encoding.reshape(-1, C)

    # 获取第一帧的绝对物理位姿 (XYZ)
    q_ref = gt_cameras.R[0]  # [4]
    T_ref = gt_cameras.T[0]  # [3] 注意：这里直接用 .T (XYZ)

    # 扩展到所有帧以便批量计算
    q_ref = q_ref.unsqueeze(0).expand(B * S, 4)
    T_ref = T_ref.unsqueeze(0).expand(B * S, 3)

    # 解码相对平移 XYZ
    delta_xyz = pose_encoding_reshaped[:, :3]
    abs_T = T_ref + delta_xyz # 直接在物理空间相加：T_curr = T_ref + ΔXYZ

    # 解码相对旋转
    delta_q = pose_encoding_reshaped[:, 3:7]
    abs_q = quaternion_multiply(delta_q, q_ref)

    # 焦距处理 (如果你的 target_dim 是 7，则没有 focal 部分，使用默认值)
    focal_length = torch.ones((B * S, 1), device=abs_q.device) * 2.0 

    # 封装回相机对象
    pred_cameras = QuaternionCameras(
        R=abs_q, 
        T=abs_T, 
        focal_length=focal_length,
        device=abs_q.device
    )
    return pred_cameras
    
def pose_encoding_to_camera2(
    pose_encoding,  # 形状: B x N x C
    pose_encoding_type="absT_quaR_OneFL",  # 编码类型
    log_focal_length_bias=1.8,
    min_focal_length=0.1,
    max_focal_length=30,
    return_dict=False,
    to_OpenCV=False,
    gt_cameras=None,
    intri_type=None, # Spark,AMD,AMD_eval,AMD_test
):
    """
    将编码向量解码为相机位姿 (R, T) 和焦距 f。

    Args:
        pose_encoding: 张量 (B, N, C)，通常 C=8：Δu, Δv, Δd, 四元数, focal
        gt_cameras: 提供参考帧（第一帧）的四元数 R 和位移 T（形状：R: (4,), T: (3,)）
    """
    B, N, C = pose_encoding.shape
    pose_encoding_reshaped = pose_encoding.reshape(-1, C)  # (B*N, C)

    # --- 获取参考帧相机位姿：四元数和位移 ---
    q_ref = gt_cameras.R[0]  # (4,)
    T_ref = gt_cameras.T_uvz[0]  # (3,)
    q_ref = q_ref.unsqueeze(0).expand(B * N, 4)  # (BN, 4)
    T_ref = T_ref.unsqueeze(0).expand(B * N, 3)  # (BN, 3)

    # --- 拆分 Δ(u,v,d) ---
    u_ref = T_ref[:, :1]  # [BN, 1]
    v_ref = T_ref[:, 1:2]
    d_ref = T_ref[:, 2:]

    if pose_encoding_type == "absT_quaR_OneFL":
        # Δu, Δv 经过归一化，恢复为像素偏移量 (单位：像素)
        delta_u = pose_encoding_reshaped[:, :1]/gt_cameras.ratio * (256 / 2)  # 宽度方向（u）
        delta_v = pose_encoding_reshaped[:, 1:2]/gt_cameras.ratio * (256 / 2)  # 高度方向（v）
        delta_d = pose_encoding_reshaped[:, 2:3]/gt_cameras.ratio  # 深度平移：单位不变（如米）

        u_abs = u_ref + delta_u  # 新的 u
        v_abs = v_ref + delta_v
        d_abs = d_ref * (delta_d + 1)  # 新的 Z 轴（深度）

        # --- 3. 处理内参 ---
        if intri_type == "spark":
            fx = 1744.92206139719
            fy = 1746.58640701753
            cx = 737.272795902663
            cy = 528.471960188736
        elif intri_type == "AMD" or intri_type == "AMD_eval":
            fx = 268.44444444
            fy = 268.44444444
            cx = 320.0
            cy = 240.0
        elif intri_type == "AMD_test":
            fx, fy = 214.75555555, 286.34074074
            cx, cy = 256.0, 256.0
        else:
            # [修正] 修复拼写错误，抛出异常
            raise ValueError(f"Unknown intri_type: {intri_type}")

        Tx = (u_abs - cx) * d_abs / fx
        Ty = (v_abs - cy) * d_abs / fy
        abs_T = torch.cat([Tx, Ty, d_abs], dim=1)  # [BN, 3] 当前帧相机在世界坐标下的位置（待旋转）

        # --- Δ四元数旋转 ---
        delta_q = pose_encoding_reshaped[:, 3:7]
        abs_q = quaternion_multiply(delta_q, q_ref)  # [BN, 4]：右乘参考四元数得到当前帧姿态

        # R = quaternion_to_matrix(R) # todo: 如果真值是旋转矩阵再将四元数转换为旋转矩阵 R
        focal_length = pose_encoding_reshaped[:, 7:8]
        focal_length = torch.clamp( # 确保焦距在 [min_focal_length, max_focal_length] 范围
            focal_length, min=min_focal_length, max=max_focal_length
        )
    else:
        raise ValueError(f"Unknown pose encoding {pose_encoding_type}")

    if to_OpenCV:
        ### From Pytorch3D coordinate to OpenCV coordinate:
        # I hate coordinate conversion
        R = abs_q.clone()
        abs_T = abs_T.clone() # 确保原始的旋转矩阵 R 和位移向量 abs_T 不会被修改。
        R[:, :, :2] *= -1
        abs_T[:, :2] *= -1 #理论上，最主要需要反转的是 Y 轴，因为 Pytorch3D 中 Y 轴指向上，而 OpenCV 中 Y 轴指向下。
        R = R.permute(0, 2, 1)

    if return_dict: # false
        return {"focal_length": focal_length, "R": R, "T": abs_T} # xyz 不是uvz

    pred_cameras = QuaternionCameras(
        focal_length=focal_length, R=abs_q, T=abs_T, device=abs_q.device
    ) # 这是一个类（通常用于处理相机模型），建立了一个 透视相机。PerspectiveCameras 是 PyTorch3D 库中的一个类，用来表示透视相机的内参和外参。
    return pred_cameras

# 输入是旋转矩阵时
# def camera_to_pose_encoding(
#     camera, # camera（包含 R、T 和 focal_length）
#     pose_encoding_type="absT_quaR_logFL",
#     log_focal_length_bias=1.8,
#     min_focal_length=0.1,
#     max_focal_length=30,
# ):
#     """
#     Inverse to pose_encoding_to_camera
#     2\包含两个焦距值（可能是fx和fy）对焦距取对数并减去偏置适用于需要精确内参优化的场景
#     3\只使用一个焦距值 直接使用原始焦距（不取对数）适用于假设fx=fy的场景
#     """
#     # 获取T=b*s
#     T = camera.R.shape[0]
#     device = camera.R.device
#     if pose_encoding_type == "absT_quaR_OneFL":  # 焦距就是不变的，幸亏保留了第一帧，要不相对焦距还不知道咋整
#         pose_encoding = torch.zeros((T, 8), device=device)
#         # 处理第一帧（参考帧）
#         pose_encoding[0, :3] = 0  # 平移为零
#         pose_encoding[0, 3:7] = torch.tensor([1, 0, 0, 0], device=camera.R.device)  # 单位四元数
#         # 处理焦距 - 所有帧保持原始焦距值（只取第一个通道）
#         focal_length = (
#                            torch.clamp(
#                                camera.focal_length,
#                                min=min_focal_length,
#                                max=max_focal_length
#                            )
#                        )[..., 0:1]  # 只取第一个通道的焦距值
#         pose_encoding[..., 7:] = focal_length  # 存储焦距信息
#
#         # 获取参考帧（第一帧）的旋转和平移
#         R_ref = camera.R[0]  # [3, 3]
#         T_ref = camera.T[0]  # [3]
#
#         # 计算其他帧相对于第一帧的变换
#         for i in range(1, T):
#             # 获取当前帧的旋转和平移
#             R_curr = camera.R[i]  # [3, 3]
#             T_curr = camera.T[i]  # [3]
#
#             # 正确的旋转矩阵乘法
#             R_relative = R_curr @ R_ref.T
#
#             # 正确的平移向量计算
#             T_relative = T_curr - (R_relative @ T_ref)
#
#             # 将相对旋转转换为四元数
#             quaternion_R_relative = matrix_to_quaternion(R_relative)
#
#             # 存储相对变换
#             pose_encoding[i, :3] = T_relative
#             pose_encoding[i, 3:7] = quaternion_R_relative
#
#     elif pose_encoding_type == "absT_quaR_logFL":
#         pose_encoding = torch.zeros(T, 9, device=device)
#
#         # 处理第一帧（参考帧）
#         pose_encoding[:, 0, :3] = 0  # 平移为零
#         pose_encoding[:, 0, 3:7] = torch.tensor([1, 0, 0, 0], device=device)  # 单位四元数
#
#         # 计算焦距的对数值（所有帧都保持原始焦距）
#         log_focal_length = (
#                 torch.log(
#                     torch.clamp(
#                         camera.focal_length,
#                         min=min_focal_length,
#                         max=max_focal_length,
#                     )
#                 ) - log_focal_length_bias
#         )
#
#         # 存储焦距信息（所有帧）
#         pose_encoding[..., 7:] = log_focal_length
#
#         # 计算其他帧相对于第一帧的变换
#         for i in range(1, T):
#             # 获取参考帧（第一帧）的旋转和平移
#             R_ref = camera.R[:, 0]  # [B, 3, 3]
#             T_ref = camera.T[:, 0]  # [B, 3]
#
#             # 获取当前帧的旋转和平移
#             R_curr = camera.R[:, i]  # [B, 3, 3]
#             T_curr = camera.T[:, i]  # [B, 3]
#
#             # 计算相对旋转: R_relative = R_curr @ R_ref.T
#             R_relative = torch.bmm(R_curr, R_ref.transpose(1, 2))
#
#             # 计算相对平移: T_relative = T_curr - R_relative @ T_ref
#             T_relative = T_curr - torch.bmm(R_relative, T_ref.unsqueeze(-1)).squeeze(-1)
#
#             # 将相对旋转转换为四元数
#             quaternion_R_relative = matrix_to_quaternion(R_relative)
#
#             # 存储相对变换
#             pose_encoding[:, i, :3] = T_relative
#             pose_encoding[:, i, 3:7] = quaternion_R_relative
#     elif pose_encoding_type == "absT_quaR":
#         # 初始化存储相对姿态的张量
#         pose_encoding = torch.zeros(T, 7, device=device)
#
#         # 处理第一帧（参考帧）
#         pose_encoding[:, 0, :3] = 0  # 平移为零
#         pose_encoding[:, 0, 3:7] = torch.tensor([1, 0, 0, 0], device=device)  # 单位四元数
#         # 获取参考帧（第一帧）的旋转和平移
#         R_ref = camera.R[:, 0]  # [B, 3, 3]
#         T_ref = camera.T[:, 0]  # [B, 3]
#
#         # 计算其他帧相对于第一帧的变换
#         for i in range(1, T):
#             # 获取当前帧的旋转和平移
#             R_curr = camera.R[:, i]  # [B, 3, 3]
#             T_curr = camera.T[:, i]  # [B, 3]
#
#             # 计算相对旋转: R_relative = R_curr @ R_ref.T
#             R_relative = torch.bmm(R_curr, R_ref.transpose(1, 2))
#
#             # 计算相对平移: T_relative = T_curr - R_relative @ T_ref
#             T_relative = T_curr - torch.bmm(R_relative, T_ref.unsqueeze(-1)).squeeze(-1)
#
#             # 将相对旋转转换为四元数
#             quaternion_R_relative = matrix_to_quaternion(R_relative)
#
#             # 存储相对变换
#             pose_encoding[:, i, :3] = T_relative
#             pose_encoding[:, i, 3:7] = quaternion_R_relative
#     else:
#         raise ValueError(f"Unknown pose encoding {pose_encoding_type}")
#
#     return pose_encoding

# 这个版本是真值是四元式的
def camera_to_pose_encoding(
        camera,
        pose_encoding_type="absT_quaR_OneFL",
        log_focal_length_bias=1.8,
        min_focal_length=0.1,
        max_focal_length=30,
):
    """
    直接使用四元数计算相对姿态
    Args:
        camera: 包含相机参数的对象
            camera.R: [T, 4] 四元数 [w, x, y, z]
            camera.T: [T, 3] 平移向量
            camera.focal_length: [T, 2] 焦距
    """
    T = camera.R.shape[0]
    device = camera.R.device

    if pose_encoding_type == "absT_quaR_OneFL":
        pose_encoding = torch.zeros((T, 8), device=device)

        # 处理第一帧
        pose_encoding[0, :3] = 0  # 平移为零
        pose_encoding[0, 3:7] = torch.tensor([1, 0, 0, 0], device=device)  # 单位四元数

        # 处理焦距
        focal_length = torch.clamp(
            camera.focal_length,
            min=min_focal_length,
            max=max_focal_length
        )[..., 0:1]
        pose_encoding[..., 7:] = focal_length

        # 获取参考帧（第一帧）的四元数和平移
        q_ref = camera.R[0]  # [4] 参考帧四元数
        T_ref = camera.T[0]  # [3] 参考帧平移

        # 计算其他帧相对于第一帧的变换
        for i in range(1, T):
            q_curr = camera.R[i]  # 当前帧四元数
            T_curr = camera.T[i]  # 当前帧平移

            # 计算相对四元数：q_relative = q_curr * q_ref^(-1)
            q_relative = quaternion_multiply(q_curr, quaternion_invert(q_ref))

            T_relative = T_curr - T_ref

            # 存储相对变换
            pose_encoding[i, :3] = T_relative
            pose_encoding[i, 3:7] = q_relative

        return pose_encoding


def camera_to_pose_encoding3(
    # 这个就是 xyz的 编码为xyz的 delta
    camera,
    pose_encoding_type="absT_quaR_OneFL",
    log_focal_length_bias=1.8,
    min_focal_length=0.1,
    max_focal_length=30,
):
    T = camera.R.shape[0]
    device = camera.R.device
    
    # 强制 target_dim 为 7 (3位 XYZ + 4位四元数)
    pose_encoding = torch.zeros((T, 7), device=device)

    # 第一帧初始化
    pose_encoding[0, :3] = 0
    pose_encoding[0, 3:7] = torch.tensor([1, 0, 0, 0], device=device)

    # 获取参考帧（第一帧）
    q_ref = camera.R[0]
    T_ref = camera.T[0] # 注意：直接使用 .T (物理坐标 XYZ)

    for i in range(1, T):
        q_curr = camera.R[i]
        T_curr = camera.T[i] # 使用物理坐标 XYZ

        # 1. 计算相对旋转 (Pizza版逻辑)
        q_relative = quaternion_multiply(q_curr, quaternion_invert(q_ref))
        
        # 2. 计算相对平移 (直接使用物理空间差异)
        # 这里建议使用差值，代表相机在物理世界中的绝对位移偏差
        T_relative = T_curr - T_ref 

        pose_encoding[i, :3] = T_relative
        pose_encoding[i, 3:7] = q_relative

    return pose_encoding


# 这个版本是真值是四元式的 融合的是pizza版本
def camera_to_pose_encoding2(
        camera,
        pose_encoding_type="absT_quaR_OneFL",
        log_focal_length_bias=1.8,
        min_focal_length=0.1,
        max_focal_length=30,
):
    """
    直接使用四元数计算相对姿态
    Args:
        camera: 包含相机参数的对象
            camera.R: [T, 4] 四元数 [w, x, y, z]
            camera.T: [T, 3] 平移向量
            camera.ratio
            camera.focal_length: [T, 2] 焦距
    """
    T = camera.R.shape[0]
    device = camera.R.device

    if pose_encoding_type == "absT_quaR_OneFL":
        pose_encoding = torch.zeros((T, 8), device=device)

        # 处理第一帧
        pose_encoding[0, :3] = 0  # 平移为零
        pose_encoding[0, 3:7] = torch.tensor([1, 0, 0, 0], device=device)  # 单位四元数

        # 处理焦距
        focal_length = torch.clamp(
            camera.focal_length,
            min=min_focal_length,
            max=max_focal_length
        )[..., 0:1]
        pose_encoding[..., 7:] = focal_length

        # 获取参考帧（第一帧）的四元数和平移
        q_ref = camera.R[0]  # [4] 参考帧四元数
        T_ref = camera.T_uvz[0]  # [3] 参考帧平移

        # 计算其他帧相对于第一帧的变换
        for i in range(1, T):
            q_curr = camera.R[i]  # 当前帧四元数
            T_curr = camera.T_uvz[i]  # 当前帧平移

            # 计算相对四元数：q_relative = q_curr * q_ref^(-1)
            q_relative = quaternion_multiply(q_curr, quaternion_invert(q_ref))

            ##############new  并进行归一化的操作#################
            delta_u = (T_curr[0] - T_ref[0])* camera.ratio/ (256/2)  # u_curr - u_ref
            delta_v = (T_curr[1] - T_ref[1] )* camera.ratio / (256 /2) # v_curr - v_ref
            delta_d = (T_curr[2] / T_ref[2]) - 1  # (depth_curr / depth_ref) - 1
            delta_d = delta_d * camera.ratio
            T_relative = torch.tensor([delta_u, delta_v, delta_d], device=T_curr.device)

            # 存储相对变换
            pose_encoding[i, :3] = T_relative
            pose_encoding[i, 3:7] = q_relative

        return pose_encoding

class SimplePoseEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim // 2)
        self.act = nn.GELU()
        self.norm1 = nn.LayerNorm(output_dim // 2)
        self.fc2 = nn.Linear(output_dim // 2, output_dim)
        self.norm2 = nn.LayerNorm(output_dim)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm1(x)
        x = self.fc2(x)
        x = self.norm2(x)
        return x

class PoseEmbedding(nn.Module):
    """用正弦-余弦（Sin-Cos）嵌入 方式来转换位姿信息，使其在高维空间中更具表示能力。这种方式常用于 位置编码（Positional Encoding），类似于 Transformer 里的位置编码"""
    def __init__(self, target_dim, n_harmonic_functions=10, append_input=True): # 768 #n_harmonic_functions = 48 target_dim输入的位姿信息维度
        super().__init__()

        # self._emb_pose = HarmonicEmbedding(
        #     n_harmonic_functions=n_harmonic_functions, append_input=append_input
        # )# 使用多少个谐波函数（默认 10）  append_input是否在嵌入后保留原始输入
        self._emb_pose = SimplePoseEmbedding(input_dim=8, output_dim=768)
        # 之后的 forward 保持和原来一样

        # self.out_dim = self._emb_pose.get_output_dim(target_dim)

    def forward(self, pose_encoding):
        e_pose_encoding = self._emb_pose(pose_encoding)
        return e_pose_encoding


def get_2d_sincos_pos_embed(
    embed_dim: int, grid_size: Union[int, Tuple[int, int]], return_grid=False
) -> torch.Tensor:
    """
    This function initializes a grid and generates a 2D positional embedding using sine and cosine functions. 生成一个基于正弦和余弦的 2D 位置编码
    It is a wrapper of get_2d_sincos_pos_embed_from_grid.
    Args:
    - embed_dim: The embedding dimension. 需要嵌入的维度
    - grid_size: The grid size.
    Returns:
    - pos_embed: The generated 2D positional embedding.
    """
    if isinstance(grid_size, tuple): # 检查 grid_size 是否是一个元组。如果是元组，表示用户提供的网格尺寸是以两个元素的形式给出的（例如，(height, width)）。
        grid_size_h, grid_size_w = grid_size
    else:
        grid_size_h = grid_size_w = grid_size
    grid_h = torch.arange(grid_size_h, dtype=torch.float) # （0-23）
    grid_w = torch.arange(grid_size_w, dtype=torch.float)
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy") # 结果grid是一个包含两个张量的元组
    grid = torch.stack(grid, dim=0) #（2，24，24）直接相连
    grid = grid.reshape([2, 1, grid_size_h, grid_size_w]) #（2，1，24，24）
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid) # 1 576 768 调用 get_2d_sincos_pos_embed_from_grid()，基于 sin 和 cos 计算位置编码。
    if return_grid: # false
        return (
            pos_embed.reshape(1, grid_size_h, grid_size_w, -1).permute(
                0, 3, 1, 2
            ),
            grid,
        )
    return pos_embed.reshape(1, grid_size_h, grid_size_w, -1).permute(
        0, 3, 1, 2
    )# （1 768 24 24）


def get_1d_sincos_pos_embed(embed_dim: int, length: int, return_grid: bool = False) -> torch.Tensor:
    """
    生成一个 1D 的正余弦位置编码。

    Args:
        embed_dim: 位置编码的维度（必须为偶数）。
        length: 时间序列的长度。
        return_grid: 是否返回生成的位置索引（默认 False）。

    Returns:
        pos_embed: 生成的 1D 位置编码，形状为 (1, embed_dim, length)。
        (如果 return_grid=True，则同时返回 grid，形状为 (1, length))
    """
    # 创建一个时间索引的向量，形状为 (length,)
    grid = torch.arange(length, dtype=torch.float)
    # 使用已有的 1D 位置编码函数生成编码，返回形状为 (1, length, embed_dim)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if return_grid:
        return pos_embed, grid.unsqueeze(0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(
    embed_dim: int, grid: torch.Tensor
) -> torch.Tensor:
    """
    This function generates a 2D positional embedding from a given grid using sine and cosine functions.
    使用了正弦（sine）和余弦（cosine）函数来编码一个给定网格的坐标
    Args:
    - embed_dim: The embedding dimension.  嵌入的维度，应该是一个偶数，因为函数将一个位置要 分成两个部分来分别处理高度和宽度的编码
    - grid: The grid to generate the embedding from. grid[0] 表示网格的高度维度（y 坐标），grid[1] 表示网格的宽度维度（x 坐标）

    Returns:
    - emb: The generated 2D positional embedding.
    """
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[0]
    )  # (H*W, D/2) 1 576 384
    emb_w = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[1]
    )  # (H*W, D/2)  1 576 384

    emb = torch.cat([emb_h, emb_w], dim=2)  # (H*W, D) 1 576 768
    return emb


def get_1d_sincos_pos_embed_from_grid(
    embed_dim: int, pos: torch.Tensor
) -> torch.Tensor:
    """
    This function generates a 1D positional embedding from a given grid using sine and cosine functions.
    从给定的位置生成一个一维的正弦余弦位置嵌入
    Args:
    - embed_dim: The embedding dimension. 应该是一个偶数，以便将其分成两个部分来分别处理正弦和余弦
    - pos: The position to generate the embedding from.

    Returns:
    - emb: The generated 1D positional embedding.
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.double, )
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb[None].float()


def get_2d_embedding(
    xy: torch.Tensor, C: int, cat_coords: bool = True
) -> torch.Tensor:
    """
    This function generates a 2D positional embedding from given coordinates using sine and cosine functions.

    Args:
    - xy: The coordinates to generate the embedding from.
    - C: The size of the embedding.
    - cat_coords: A flag to indicate whether to concatenate the original coordinates to the embedding.

    Returns:
    - pe: The generated 2D positional embedding.
    """
    B, N, D = xy.shape
    assert D == 2

    x = xy[:, :, 0:1]
    y = xy[:, :, 1:2]
    div_term = (
        torch.arange(0, C, 2, device=xy.device, dtype=torch.float32)
        * (1000.0 / C)
    ).reshape(1, 1, int(C / 2))

    pe_x = torch.zeros(B, N, C, device=xy.device, dtype=torch.float32)
    pe_y = torch.zeros(B, N, C, device=xy.device, dtype=torch.float32)

    pe_x[:, :, 0::2] = torch.sin(x * div_term)
    pe_x[:, :, 1::2] = torch.cos(x * div_term)

    pe_y[:, :, 0::2] = torch.sin(y * div_term)
    pe_y[:, :, 1::2] = torch.cos(y * div_term)

    pe = torch.cat([pe_x, pe_y], dim=2)  # (B, N, C*3)
    if cat_coords:
        pe = torch.cat([xy, pe], dim=2)  # (B, N, C*3+3)
    return pe


def bilinear_sampler(input, coords, align_corners=True, padding_mode="border"):
    r"""Sample a tensor using bilinear interpolation

    `bilinear_sampler(input, coords)` samples a tensor :attr:`input` at
    coordinates :attr:`coords` using bilinear interpolation. It is the same
    as `torch.nn.functional.grid_sample()` but with a different coordinate
    convention.

    The input tensor is assumed to be of shape :math:`(B, C, H, W)`, where
    :math:`B` is the batch size, :math:`C` is the number of channels,
    :math:`H` is the height of the image, and :math:`W` is the width of the
    image. The tensor :attr:`coords` of shape :math:`(B, H_o, W_o, 2)` is
    interpreted as an array of 2D point coordinates :math:`(x_i,y_i)`.

    Alternatively, the input tensor can be of size :math:`(B, C, T, H, W)`,
    in which case sample points are triplets :math:`(t_i,x_i,y_i)`. Note
    that in this case the order of the components is slightly different
    from `grid_sample()`, which would expect :math:`(x_i,y_i,t_i)`.

    If `align_corners` is `True`, the coordinate :math:`x` is assumed to be
    in the range :math:`[0,W-1]`, with 0 corresponding to the center of the
    left-most image pixel :math:`W-1` to the center of the right-most
    pixel.

    If `align_corners` is `False`, the coordinate :math:`x` is assumed to
    be in the range :math:`[0,W]`, with 0 corresponding to the left edge of
    the left-most pixel :math:`W` to the right edge of the right-most
    pixel.

    Similar conventions apply to the :math:`y` for the range
    :math:`[0,H-1]` and :math:`[0,H]` and to :math:`t` for the range
    :math:`[0,T-1]` and :math:`[0,T]`.

    Args:
        input (Tensor): batch of input images.
        coords (Tensor): batch of coordinates.
        align_corners (bool, optional): Coordinate convention. Defaults to `True`.
        padding_mode (str, optional): Padding mode. Defaults to `"border"`.

    Returns:
        Tensor: sampled points.
    """

    sizes = input.shape[2:]

    assert len(sizes) in [2, 3]

    if len(sizes) == 3:
        # t x y -> x y t to match dimensions T H W in grid_sample
        coords = coords[..., [1, 2, 0]]

    if align_corners:
        coords = coords * torch.tensor(
            [2 / max(size - 1, 1) for size in reversed(sizes)],
            device=coords.device,
        )
    else:
        coords = coords * torch.tensor(
            [2 / size for size in reversed(sizes)], device=coords.device
        )

    coords -= 1

    return F.grid_sample(
        input, coords, align_corners=align_corners, padding_mode=padding_mode
    )


def sample_features4d(input, coords):
    r"""Sample spatial features

    `sample_features4d(input, coords)` samples the spatial features
    :attr:`input` represented by a 4D tensor :math:`(B, C, H, W)`.

    The field is sampled at coordinates :attr:`coords` using bilinear
    interpolation. :attr:`coords` is assumed to be of shape :math:`(B, R,
    2)`, where each sample has the format :math:`(x_i, y_i)`. This uses the
    same convention as :func:`bilinear_sampler` with `align_corners=True`.

    The output tensor has one feature per point, and has shape :math:`(B,
    R, C)`.

    Args:
        input (Tensor): spatial features.
        coords (Tensor): points.

    Returns:
        Tensor: sampled features.
    """

    B, _, _, _ = input.shape

    # B R 2 -> B R 1 2
    coords = coords.unsqueeze(2)

    # B C R 1
    feats = bilinear_sampler(input, coords)

    return feats.permute(0, 2, 1, 3).view(
        B, -1, feats.shape[1] * feats.shape[3]
    )  # B C R 1 -> B R C
