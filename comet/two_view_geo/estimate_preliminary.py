# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from minipytorch3d.cameras import PerspectiveCameras

# from pytorch3d.renderer.cameras import CamerasBase, PerspectiveCameras
# from pytorch3d.transforms import se3_exp_map, se3_log_map, Transform3d, so3_relative_angle


from torch.cuda.amp import autocast

from .fundamental import estimate_fundamental, essential_from_fundamental
from .homography import estimate_homography, decompose_homography_matrix
from .essential import estimate_essential, decompose_essential_matrix
from .utils import get_default_intri, remove_cheirality

# TODO remove the .. and . that may look confusing
from ..utils.metric import closed_form_inverse

try:
    import poselib

    print("Poselib is available")
except:
    print("Poselib is not installed. Please disable use_poselib")


def estimate_preliminary_cameras_poselib(
    tracks,
    tracks_vis,
    width,
    height,
    tracks_score=None,
    max_error=0.5,
    max_ransac_iters=20000,
    predict_essential=False,
    lo_num=None,
    predict_homo=False,
    loopresidual=False,
):
    B, S, N, _ = tracks.shape

    query_points = tracks[:, 0:1].reshape(B, N, 2)
    reference_points = tracks[:, 1:].reshape(B * (S - 1), N, 2)

    valid_mask = (tracks_vis >= 0.05)[:, 1:].reshape(B * (S - 1), N)

    fmat = []
    inlier_mask = []
    for idx in range(len(reference_points)):
        kps_left = query_points[0].cpu().numpy()
        kps_right = reference_points[idx].cpu().numpy()

        cur_inlier_mask = valid_mask[idx].cpu().numpy()

        kps_left = kps_left[cur_inlier_mask]
        kps_right = kps_right[cur_inlier_mask]

        cur_fmat, info = poselib.estimate_fundamental(
            kps_left,
            kps_right,
            {
                "max_epipolar_error": max_error,
                "max_iterations": max_ransac_iters,
                "min_iterations": 1000,
                "real_focal_check": True,
                "progressive_sampling": False,
            },
        )

        cur_inlier_mask[cur_inlier_mask] = np.array(info["inliers"])

        fmat.append(cur_fmat)
        inlier_mask.append(cur_inlier_mask)

    fmat = torch.from_numpy(np.array(fmat)).to(query_points.device)
    inlier_mask = torch.from_numpy(np.array(inlier_mask)).to(
        query_points.device
    )

    preliminary_dict = {
        "fmat": fmat[None],
        "fmat_inlier_mask": inlier_mask[None],
    }

    return None, preliminary_dict


def estimate_preliminary_cameras(
    tracks,
    tracks_vis,
    width,
    height,
    tracks_score=None,
    max_error=0.5,
    lo_num=300,
    max_ransac_iters=4096,
    predict_essential=False,
    predict_homo=False,
    loopresidual=False,
):
    # TODO: also clean the code for predict_essential and predict_homo

    with autocast(dtype=torch.double): # 相对于第一帧
        # batch_num, frame_num, point_num
        B, S, N, _ = tracks.shape # 1 8 8192 2

        # We have S-1 reference frame per batch 这些操作将每个批次中从第二帧开始，每一帧的点与第一帧的点配对，形成用于几何估计的对应关系。
        query_points = ( #7 8192 2 查询点（query_points）：出每个批次中第一帧（索引 0）的跟踪点，固定的查询帧。复制S次 -》(B * (S - 1), N, 2)
            tracks[:, 0:1].expand(-1, S - 1, -1, -1).reshape(B * (S - 1), N, 2)
        )
        reference_points = tracks[:, 1:].reshape(B * (S - 1), N, 2) #7 8192 2参考点（reference_points）： 取出除了第一帧以外的所有帧中跟踪点的信息。重塑为 (B*(S-1), N, 2)，这样每个参考帧都对应一组 N 个 2D 点。

        # Filter out some matches based on track vis and score 基于可见性和匹配得分筛选有效匹配点 只对参考帧（从第二帧开始）进行判断，
        # 因为查询帧（第一帧）的点通常默认都是有效的。其中每一行对应一个参考帧中的 N 个点是否有效。
        valid_mask = (tracks_vis >= 0.05)[:, 1:].reshape(B * (S - 1), N) #7 8192

        if tracks_score is not None:
            valid_tracks_score_mask = (tracks_score >= 0.5)[:, 1:].reshape(
                B * (S - 1), N
            ) #7 8192
            valid_mask = torch.logical_and(valid_mask, valid_tracks_score_mask) #7 8192 只有同时满足可见性和得分要求的匹配才认为是有效的。

        # Estimate Fundamental Matrix by Batch 利用上面构造的点对（query_points 和 reference_points）和有效掩码（valid_mask）来批量估计基础矩阵
        # fmat: (B*(S-1))x3x3
        (fmat, fmat_inlier_num, fmat_inlier_mask, fmat_residuals) = (# 7 3 3，7 ，7 8192， 7 8192
            estimate_fundamental(
                query_points, # 一对的左边
                reference_points, # 一对的左边
                max_error=max_error, #4 最大误差阈值
                lo_num=lo_num, #300 局部精炼时使用的最小内点数量
                max_ransac_iters=max_ransac_iters, #4096 RANSAC迭代次数
                valid_mask=valid_mask, # 有效匹配的掩码
                loopresidual=loopresidual, #true
                return_residuals=True,
            )
        )

        # kmat1, kmat2一模一样: (B*(S-1))x3x3  S-1 4 ，S-1 4
        kmat1, kmat2, fl, pp = build_default_kmat(
            width,
            height,
            B,
            S,
            N,
            device=query_points.device,
            dtype=query_points.dtype,
        )

        emat_fromf, _, _ = essential_from_fundamental(fmat, kmat1, kmat2) #7 3 3 利用相机内参矩阵 kmat1 和 kmat2 将基础矩阵 fmat 转换为本质矩阵（essential matrix）

        R_emat_fromf, t_emat_fromf = decompose_essential_matrix(emat_fromf) # 将本质矩阵分解成旋转矩阵 R_emat_fromf 7 候选解数：4 3 3 和平移向量 t_emat_fromf 7 4 3
        R_emat_fromf, t_emat_fromf = remove_cheirality( #733 73 利用正向性（cheirality）条件选择出正确的候选解。也就是说，只有满足场景中物体应出现在相机前方的解才被保留。
            R_emat_fromf, t_emat_fromf, query_points, reference_points, fl, pp
        ) # query_points 和 reference_points 提供了点对应信息，fl 和 pp（焦距与主点）用于辅助验证。

        # TODO: clean the code for R_hmat, t_hmat, R_emat, t_emat and add them here
        R_preliminary = R_emat_fromf # 直接把经过 cheirality 筛选后的旋转和平移解作为初步的相机位姿
        t_preliminary = t_emat_fromf

        R_preliminary = R_preliminary.reshape(B, S - 1, 3, 3)
        t_preliminary = t_preliminary.reshape(B, S - 1, 3)

        # pad for the first camera 由于第一帧作为参考帧，其位姿应被设为默认值（旋转为单位矩阵，平移为零）
        R_pad = (
            torch.eye(3, device=tracks.device, dtype=tracks.dtype)[None]
            .repeat(B, 1, 1)
            .unsqueeze(1)
        )
        t_pad = (
            torch.zeros(3, device=tracks.device, dtype=tracks.dtype)[None]
            .repeat(B, 1)
            .unsqueeze(1)
        )
        #  将第一帧的位姿与剩余帧的位姿拼接起来
        R_preliminary = torch.cat([R_pad, R_preliminary], dim=1).reshape(
            B * S, 3, 3
        )
        t_preliminary = torch.cat([t_pad, t_preliminary], dim=1).reshape(
            B * S, 3
        )
        # 将初步位姿数据复制一份
        R_opencv = R_preliminary.clone()
        t_opencv = t_preliminary.clone()

        # From OpenCV/COLMAP camera convention to PyTorch3D OpenCV 和 COLMAP 通常采用的坐标系与 PyTorch3D 默认使用的坐标系不完全一致，需要做一定的转换
        # TODO: Remove the usage of PyTorch3D convention in all the codebase
        # So that we don't need to do such conventions any more
        R_preliminary = R_preliminary.clone().permute(0, 2, 1)
        t_preliminary = t_preliminary.clone()
        t_preliminary[:, :2] *= -1
        R_preliminary[:, :, :2] *= -1

        pred_cameras = PerspectiveCameras( # 用经过转换后的旋转矩阵 R 和平移向量 T 构造 PyTorch3D 中的透视相机对象
            R=R_preliminary, T=t_preliminary, device=R_preliminary.device
        ) # 8个相机

        with autocast(dtype=torch.double): # 归一化相机位姿：使所有相机参数相对于第一帧（参考帧），通过计算世界到视图变换、求逆、批量矩阵乘法以及调整齐次矩阵来完成
            # Optional in the future 将相机参数调整为相对于第一帧（全局归一化）
            # make all the cameras relative to the first one
            pred_se3 = pred_cameras.get_world_to_view_transform().get_matrix() #8个 Transform3d 对象对应的matrix，外参矩阵8 4 4

            rel_transform = closed_form_inverse(pred_se3[0:1, :, :]) #1 4 4 求逆 第一帧从相机到世界的变换 有错误 后面看
            rel_transform = rel_transform.expand(B * S, -1, -1) # 将该逆矩阵扩展到所有相机

            pred_se3_rel = torch.bmm(rel_transform, pred_se3) #8 4 4 通过批量矩阵乘法 torch.bmm(rel_transform, pred_se3) 得到所有相机相对于第一帧的变换
            pred_se3_rel[..., :3, 3] = 0.0 #8 4 4 将齐次矩阵中的平移部分（前三行第4列）强制置为 0。这里不是平移部分 是旋转矩阵的最后一行而已
            pred_se3_rel[..., 3, 3] = 1.0 # ，同时保证右下角为 1

            pred_cameras.R = pred_se3_rel[:, :3, :3].clone() # 从更新后的 4×4 矩阵中提取旋转部分（pred_se3_rel[:, :3, :3]）并赋值给 pred_cameras.R
            pred_cameras.T = pred_se3_rel[:, 3, :3].clone()

        # Record them in case we may need later
        fmat = fmat.reshape(B, S - 1, 3, 3)
        fmat_inlier_mask = fmat_inlier_mask.reshape(B, S - 1, -1)
        kmat1 = kmat1.reshape(B, S - 1, 3, 3)
        R_opencv = R_opencv.reshape(B, S, 3, 3)
        t_opencv = t_opencv.reshape(B, S, 3)

        fmat_residuals = fmat_residuals.reshape(B, S - 1, -1)

        preliminary_dict = {
            "fmat": fmat, # 1 7 3 3基础矩阵
            "fmat_inlier_mask": fmat_inlier_mask,# 1 7 8192 内点掩码
            "R_opencv": R_opencv, # 1 8 3 3
            "t_opencv": t_opencv,# 183
            "default_intri": kmat1, # 默认的内参
            "emat_fromf": emat_fromf, #7 3 3 本质矩阵
            "fmat_residuals": fmat_residuals, #1 7 8192 误差
        }

        return pred_cameras, preliminary_dict


def build_default_kmat(width, height, B, S, N, device=None, dtype=None):
    # focal length is set as max(width, height) 构造两个默认的相机内参矩阵（kmat1 和 kmat2），
    # principal point is set as (width//2, height//2,) 以及对应的焦距（fl）和主点（pp）的参数，用于左右图像（或左右帧）的相机标定

    fl, pp, _ = get_default_intri(width, height, device, dtype) # 用 get_default_intri 函数，根据图像的宽度和高度生成默认的内参
    # fl通常设置为 max(width, height)*ratio，保证较大的尺寸。 主点（pp）：设置为图像中心，即 (width//2, height//2).
    # :2 for left frame, 2: for right frame
    fl = torch.ones((B, S - 1, 4), device=device, dtype=dtype) * fl # 1 S-1 4对参考帧（左右帧对），每个对需要4个数，左图和右图的xy方向的焦距
    pp = torch.cat([pp, pp])[None][None].expand(B, S - 1, -1) # 1 7 4 拼接成包含两个主点的向量（例如 [cx, cy, cx, cy]），表示左右图像使用相同的主点

    fl = fl.reshape(B * (S - 1), 4) # 7 4每一行对应一对左右图像（或左右帧）的参数，其中前两个数字表示左侧图像的焦距或主点，后两个数字表示右侧图像的焦距或主点。
    pp = pp.reshape(B * (S - 1), 4)

    # build kmat (B*(S-1), 3, 3)
    kmat1 = torch.eye(3, device=device, dtype=dtype)[None].repeat(
        B * (S - 1), 1, 1
    )
    kmat2 = torch.eye(3, device=device, dtype=dtype)[None].repeat(
        B * (S - 1), 1, 1
    )

    # assign them to the corresponding locations of kmats
    kmat1[:, [0, 1], [0, 1]] = fl[:, :2]
    kmat1[:, [0, 1], 2] = pp[:, :2]

    kmat2[:, [0, 1], [0, 1]] = fl[:, 2:]
    kmat2[:, [0, 1], 2] = pp[:, 2:]

    return kmat1, kmat2, fl, pp
