# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import math
from scipy.spatial.transform import Rotation as R
import random
import numpy as np
import torch
import torch.nn.functional as F

from pytorch3d.transforms import random_quaternions
from torch.nn import MSELoss

from minipytorch3d.rotation_conversions import (
    matrix_to_quaternion,
    quaternion_to_matrix,
)


def quaternion_from_matrix(matrix, isprecise=False):
    """Return quaternion from rotation matrix.

    If isprecise is True, the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.

    >>> q = quaternion_from_matrix(numpy.identity(4), True)
    >>> numpy.allclose(q, [1, 0, 0, 0])
    True
    >>> q = quaternion_from_matrix(numpy.diag([1, -1, -1, 1]))
    >>> numpy.allclose(q, [0, 1, 0, 0]) or numpy.allclose(q, [0, -1, 0, 0])
    True
    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R, True)
    >>> numpy.allclose(q, [0.9981095, 0.0164262, 0.0328524, 0.0492786])
    True
    >>> R = [[-0.545, 0.797, 0.260, 0], [0.733, 0.603, -0.313, 0],
    ...      [-0.407, 0.021, -0.913, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.19069, 0.43736, 0.87485, -0.083611])
    True
    >>> R = [[0.395, 0.362, 0.843, 0], [-0.626, 0.796, -0.056, 0],
    ...      [-0.677, -0.498, 0.529, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.82336615, -0.13610694, 0.46344705, -0.29792603])
    True
    >>> R = random_rotation_matrix()
    >>> q = quaternion_from_matrix(R)
    >>> is_same_transform(R, quaternion_matrix(q))
    True
    >>> R = euler_matrix(0.0, 0.0, numpy.pi/2.0)
    >>> numpy.allclose(quaternion_from_matrix(R, isprecise=False),
    ...                quaternion_from_matrix(R, isprecise=True))
    True

    """

    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4,))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]

        # symmetric matrix K
        K = np.array(
            [
                [m00 - m11 - m22, 0.0, 0.0, 0.0],
                [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
            ]
        )
        K /= 3.0

        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]

    if q[0] < 0.0:
        np.negative(q, q)

    return q


def euler_angle_error_deg(pred_euler: torch.Tensor, gt_euler: torch.Tensor) -> torch.Tensor:
    """
    计算逐轴欧拉角误差（角度制，考虑周期）
    Args:
        pred_euler: (N, 3) 预测欧拉角
        gt_euler: (N, 3) 真值欧拉角
    Returns:
        angle_error: (N, 3) 欧拉角误差，每轴 ∈ [0, 180]
    """
    delta = pred_euler - gt_euler
    delta = (delta + 180) % 360 - 180  # wrap to [-180, 180]
    return delta.abs()


def quaternion_wxyz_to_euler_deg(q: torch.Tensor, order="xyz") -> torch.Tensor:
    """
    将四元数（[w, x, y, z]）转换为欧拉角（角度制）
    Args:
        q: (N, 4) tensor, 四元数格式为 [w, x, y, z]
        order: 欧拉角顺序，如 'xyz', 'zyx'
    Returns:
        euler_deg: (N, 3) 欧拉角，单位为度
    """
    q = q.detach().cpu().numpy()  # (N, 4)
    q_xyzw = np.concatenate([q[:, 1:], q[:, 0:1]], axis=1)  # 变成 [x, y, z, w]
    euler_deg = R.from_quat(q_xyzw).as_euler(order, degrees=True)
    return torch.from_numpy(euler_deg).float()

def camera_to_rel_deg(pred_cameras, gt_cameras, device, batch_size):
    """
    Calculate relative rotation and translation angles between predicted and ground truth cameras.
    Args:
    - pred_cameras: Predicted camera.
    - gt_cameras: Ground truth camera.
    - accelerator: The device for moving tensors to GPU or others.
    - batch_size: Number of data samples in one batch.

    Returns:
    - rel_rotation_angle_deg, rel_translation_angle_deg: Relative rotation and translation angles in degrees.
    """

    with torch.no_grad():
        # 1) 将相机对象转换为 4x4 的 SE(3) 变换矩阵，形状均为 [B*S,4,4]
        gt_se3 = gt_cameras.get_world_to_view_transform().get_matrix()
        pred_se3 = pred_cameras.get_world_to_view_transform().get_matrix()

        # 2) 按序列（batch）内部所有帧两两配对：生成两个索引数组 pair_idx_i1, pair_idx_i2
        #    每个数组长度 = batch_size * (frames_per_seq choose 2)
        pair_idx_i1, pair_idx_i2 = batched_all_pairs(
            batch_size, gt_se3.shape[0] // batch_size
        )
        pair_idx_i1 = pair_idx_i1.to(device)

        ##########################his method##########################
        # 3) 计算真值的相对位姿：T_gt_rel = (T_gt_i)^(-1) * T_gt_j
        # 4) 同理计算预测的相对位姿
        relative_pose_gt = closed_form_inverse(gt_se3[pair_idx_i1]).bmm(
            gt_se3[pair_idx_i2]
        )
        relative_pose_pred = closed_form_inverse(pred_se3[pair_idx_i1]).bmm(
            pred_se3[pair_idx_i2]
        )
        rel_rangle_deg1 = rotation_angle(
            relative_pose_gt[:, :3, :3], relative_pose_pred[:, :3, :3]
        )
        rel_tangle_deg1 = translation_angle(
            relative_pose_gt[:, 3, :3], relative_pose_pred[:, 3, :3]
        )
    return rel_rangle_deg1, rel_tangle_deg1


def camera_to_rel_deg3(pred_cameras, gt_cameras, device, batch_size):
    """
    Calculate relative rotation and translation angles between predicted and ground truth cameras.
    Args:
    - pred_cameras: Predicted camera.
    - gt_cameras: Ground truth camera.
    - accelerator: The device for moving tensors to GPU or others.
    - batch_size: Number of data samples in one batch.

    Returns:
    rel_rangle_deg1: rotation angle error (deg) between relative poses
    rel_tangle_deg1: translation direction angle error (deg)
    translation_err: absolute translation RMSE
    X_err, Y_err, Z_err: per-axis absolute translation RMSE
    """

    with torch.no_grad():
        ########################new######################
        # we do not take into account the first frame, this is why translations_gt->translations_gt[:, 1:, :]
        # 用 sum 版本
        translation_err = F.mse_loss(pred_cameras.T, gt_cameras.T, reduction='sum')
        X_err = F.mse_loss(pred_cameras.T[:, 0], gt_cameras.T[:, 0], reduction='sum')
        Y_err = F.mse_loss(pred_cameras.T[:, 1], gt_cameras.T[:, 1], reduction='sum')
        Z_err = F.mse_loss(pred_cameras.T[:, 2], gt_cameras.T[:, 2], reduction='sum')
        # print(pred_cameras.T)
        # print(gt_cameras.T)

        # as the loss_function is defined with reduction="sum" and **2, we have to rescale it to get the correct mean
        N_elements = pred_cameras.T.shape[0]  # num_elements in loss function
        translation_err = torch.sqrt(translation_err / N_elements) * (10 ** 3)
        X_err = torch.sqrt(X_err / N_elements)* (10 ** 3)
        Y_err = torch.sqrt(Y_err / N_elements)* (10 ** 3)
        Z_err = torch.sqrt(Z_err / N_elements)* (10 ** 3)
        ########################new######################

        # 1) 将相机对象转换为 4x4 的 SE(3) 变换矩阵，形状均为 [B*S,4,4]
        gt_se3 = gt_cameras.get_world_to_view_transform().get_matrix()
        # print(gt_se3)
        pred_se3 = pred_cameras.get_world_to_view_transform().get_matrix()

        # 2) 按序列（batch）内部所有帧两两配对：生成两个索引数组 pair_idx_i1, pair_idx_i2
        #    每个数组长度 = batch_size * (frames_per_seq choose 2)
        pair_idx_i1, pair_idx_i2 = batched_all_pairs(
            batch_size, gt_se3.shape[0] // batch_size
        )
        pair_idx_i1 = pair_idx_i1.to(device)
        # print(gt_se3[0])

        ##########################his method##########################
        # 3) 计算真值的相对位姿：T_gt_rel = (T_gt_i)^(-1) * T_gt_j
        # 4) 同理计算预测的相对位姿
        relative_pose_gt = closed_form_inverse(gt_se3[pair_idx_i1]).bmm(
            gt_se3[pair_idx_i2]
        )
        relative_pose_pred = closed_form_inverse(pred_se3[pair_idx_i1]).bmm(
            pred_se3[pair_idx_i2]
        )
        rel_rangle_deg1 = rotation_angle(
            relative_pose_gt[:, :3, :3], relative_pose_pred[:, :3, :3]
        )
        rel_tangle_deg1 = translation_angle(
            relative_pose_gt[:, 3, :3], relative_pose_pred[:, 3, :3]
        )

    return rel_rangle_deg1, rel_tangle_deg1, translation_err, X_err, Y_err, Z_err

def camera_to_rel_deg2(pred_pose_enc, gt_enc, device, batch_size, with_cumulative_err=False):
    """
    计算预测相机位姿与真值之间的旋转和平移角度误差。并计算角度的平均值， 然后输出欧拉角误差 这就是pizza的方法
    返回值:
    - rel_rangle_deg: [B*S] 每帧的相对旋转角误差（通过四元数的 geodesic 距离计算，单位为度）
    - rel_tangle_deg: [B*S] 每帧的相对平移方向角误差（两个平移向量之间的夹角，单位为度）
    - tracking_geodesic_err: 所有帧预测旋转与真值之间的平均 geodesic 角误差（单位为度）
    - tracking_error_euler: [3,] 欧拉角绝对误差的均值（分别表示绕 X、Y、Z 轴的误差，单位为度）
    - cumulative_err: [S-1,] 每个时间步的累积旋转误差（仅当 with_cumulative_err=True 时返回，单位为度）
    """

    with torch.no_grad():
        # q_gt = gt_enc[:, 3:7]
        # q_pred = pred_pose_enc[:, 3:7]
        # q_gt = q_gt / q_gt.norm(dim=1, keepdim=True)
        # q_pred = q_pred / q_pred.norm(dim=1, keepdim=True)
        # rel_rangle_deg = rotation_angle2(q_gt, q_pred)
        # # 2. 欧拉角转换
        # euler_gt = quaternion_to_euler_angles(q_gt, degrees=True)  # [B, 3]
        # euler_pred = quaternion_to_euler_angles(q_pred, degrees=True)  # [B, 3]
        # # 3. 计算绝对误差
        # euler_error = torch.abs(euler_gt - euler_pred)  # [B, 3]
        # print("hh,",euler_error)
        # print("hhh")

        rel_tangle_deg = translation_angle(
            gt_enc[:, :3], pred_pose_enc[:, :3]
        )

        rel_pred_mat = quaternion_to_matrix(pred_pose_enc[..., 3:7])
        rel_gt_mat = quaternion_to_matrix(gt_enc[..., 3:7])


        # get the tracking geodesic distance between prediction and ground-truth
        geodesic_err, error_euler = geodesic_distance_from_two_batches(
            rel_pred_mat.reshape(-1, 3, 3),
            rel_gt_mat.reshape(-1, 3, 3),
            with_euler_error=True)
        rel_rangle_deg = torch.rad2deg(geodesic_err)
        error_euler = np.rad2deg(error_euler)
        avg_rangle_deg = rel_rangle_deg.mean() # 就是取平均值而已

        if with_cumulative_err:
            len_sequences_minus  = gt_enc.shape[0] // batch_size
            cumulative_err = geodesic_err.reshape(batch_size, len_sequences_minus)  # reshape from (Bx(L-1)) to
            # Bx(L-1)
            cumulative_err = torch.mean(cumulative_err, dim=0)  # get error w.r.t temporal frame -> (L-1)
            cumulative_err = torch.cumsum(cumulative_err, dim=0)  # get cumulative error
            cumulative_err = torch.rad2deg(cumulative_err)
        else:
            cumulative_err = None
    return rel_rangle_deg, rel_tangle_deg, avg_rangle_deg, error_euler, cumulative_err

def rotationMatrixToEulerAngles(batch_matrix_rotation):
    """
    :param batch_matrix_rotation: Bx3x3 list matrix of rotation
    :return: euler angles corresponding to each matrix
    """
    batch_size = len(batch_matrix_rotation)
    error_euler = np.zeros((batch_size, 3))
    for i in range(batch_size):
        R = batch_matrix_rotation[i]
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

        singular = sy < 1e-6

        if not singular:
            z = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            x = math.atan2(R[1, 0], R[0, 0])
        else:
            z = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            x = 0
        error_euler[i] = [x, y, z]
    return error_euler

def geodesic_distance_from_two_batches(batch1, batch2, with_euler_error=True):
    """

    :param batch1: Bx3x3 B matrix of rotation
    :param batch2: Bx3x3 B matrix of rotation
    :param with_euler_error: whether output euler error for each axis
    :return: B theta angles of the matrix rotation from batch1 to batch2
    """
    batch = batch1.shape[0]
    m = torch.bmm(batch1, batch2.transpose(1, 2))  # batch*3*3
    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).cuda()))
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).cuda()) * -1)
    theta = torch.acos(cos)

    if with_euler_error:
        m = m.cpu().detach().numpy()
        euler_angles = rotationMatrixToEulerAngles(m)
        error_euler = np.mean(np.abs(euler_angles), axis=0)
    else:
        error_euler = None
    return theta, error_euler,euler_angles


def quaternion_to_euler_angles(quat_batch: torch.Tensor, degrees=True):
    """
    将四元数 (WXYZ) 批量转换为欧拉角 (XYZ顺序)

    Args:
        quat_batch: Tensor of shape [B, 4], 格式为 WXYZ
        degrees: 是否返回角度单位（True -> degree）

    Returns:
        euler_angles: Tensor of shape [B, 3], 每行为 [X, Y, Z] 欧拉角
    """
    # 转换为 numpy 并调整为 XYZW 顺序，因为 scipy 接受的是 XYZW
    quat_np = quat_batch[:, [1, 2, 3, 0]].cpu().numpy()
    r = R.from_quat(quat_np)
    euler = r.as_euler('xyz', degrees=degrees)  # XYZ顺序
    return torch.from_numpy(euler).to(quat_batch.device)  # 保持设备一致


def geodesic_distance_from_two_batches2(batch1, batch2, with_euler_error=True):
    """

    :param batch1: Bx3x3 B matrix of rotation
    :param batch2: Bx3x3 B matrix of rotation
    :param with_euler_error: whether output euler error for each axis
    :return: B theta angles of the matrix rotation from batch1 to batch2
    """
    batch = batch1.shape[0]
    m = torch.bmm(batch1, batch2.transpose(1, 2))  # batch*3*3
    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).cuda()))
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).cuda()) * -1)
    theta = torch.acos(cos)

    if with_euler_error:
        m = m.cpu().detach().numpy()
        euler_angles = rotationMatrixToEulerAngles(m)
    else:
        euler_angles = None
    return theta, euler_angles


def camera_to_rel_deg2(pred_pose_enc, gt_enc, device, batch_size, with_cumulative_err=False):
    """
    计算预测相机位姿与真值之间的旋转和平移角度误差。并计算角度的平均值， 然后输出欧拉角误差 这就是pizza的方法
    返回值:
    - rel_rangle_deg: [B*S] 每帧的相对旋转角误差（通过四元数的 geodesic 距离计算，单位为度）
    - rel_tangle_deg: [B*S] 每帧的相对平移方向角误差（两个平移向量之间的夹角，单位为度）
    - tracking_geodesic_err: 所有帧预测旋转与真值之间的平均 geodesic 角误差（单位为度）
    - tracking_error_euler: [3,] 欧拉角绝对误差的均值（分别表示绕 X、Y、Z 轴的误差，单位为度）
    - cumulative_err: [S-1,] 每个时间步的累积旋转误差（仅当 with_cumulative_err=True 时返回，单位为度）
    """

    with torch.no_grad():
        # q_gt = gt_enc[:, 3:7]
        # q_pred = pred_pose_enc[:, 3:7]
        # q_gt = q_gt / q_gt.norm(dim=1, keepdim=True)
        # q_pred = q_pred / q_pred.norm(dim=1, keepdim=True)
        # rel_rangle_deg = rotation_angle2(q_gt, q_pred)
        # # 2. 欧拉角转换
        # euler_gt = quaternion_to_euler_angles(q_gt, degrees=True)  # [B, 3]
        # euler_pred = quaternion_to_euler_angles(q_pred, degrees=True)  # [B, 3]
        # # 3. 计算绝对误差
        # euler_error = torch.abs(euler_gt - euler_pred)  # [B, 3]
        # print("hh,",euler_error)
        # print("hhh")

        rel_tangle_deg = translation_angle(
            gt_enc[:, :3], pred_pose_enc[:, :3]
        )

        rel_pred_mat = quaternion_to_matrix(pred_pose_enc[..., 3:7])
        rel_gt_mat = quaternion_to_matrix(gt_enc[..., 3:7])


        # get the tracking geodesic distance between prediction and ground-truth
        geodesic_err, error_euler, error_eulers = geodesic_distance_from_two_batches(
            rel_pred_mat.reshape(-1, 3, 3),
            rel_gt_mat.reshape(-1, 3, 3),
            with_euler_error=True)
        rel_rangle_deg = torch.rad2deg(geodesic_err)
        error_euler = np.rad2deg(error_euler)
        error_eulers = np.rad2deg(error_eulers)
        avg_rangle_deg = rel_rangle_deg.mean() # 就是取平均值而已

        ############nrew ############
        # error_euler: (N, 3) 角度误差 (单位: deg)
        threshold = 5.0
        # 逐列统计 <5° 的比例
        percentages = (error_eulers < threshold).mean(axis=0)
        # 转成列表 [x%, y%, z%]
        percentages_list = percentages.tolist()

        # if with_cumulative_err:
        #     len_sequences_minus  = gt_enc.shape[0] // batch_size
        #     cumulative_err = geodesic_err.reshape(batch_size, len_sequences_minus)  # reshape from (Bx(L-1)) to
        #     # Bx(L-1)
        #     cumulative_err = torch.mean(cumulative_err, dim=0)  # get error w.r.t temporal frame -> (L-1)
        #     cumulative_err = torch.cumsum(cumulative_err, dim=0)  # get cumulative error
        #     cumulative_err = torch.rad2deg(cumulative_err)
        # else:
        #     cumulative_err = None
    return rel_rangle_deg, rel_tangle_deg, avg_rangle_deg, error_euler, percentages_list


def camera_to_rel_deg4(pred_pose_enc, gt_enc, device, batch_size, with_cumulative_err=False):
    """
    计算预测相机位姿与真值之间的旋转和平移角度误差。并计算角度的平均值， 然后输出欧拉角误差 这就是pizza的方法
    返回值:
    - rel_rangle_deg: [B*S] 每帧的相对旋转角误差（通过四元数的 geodesic 距离计算，单位为度）
    - rel_tangle_deg: [B*S] 每帧的相对平移方向角误差（两个平移向量之间的夹角，单位为度）
    - tracking_geodesic_err: 所有帧预测旋转与真值之间的平均 geodesic 角误差（单位为度）
    - tracking_error_euler: [3,] 欧拉角绝对误差的均值（分别表示绕 X、Y、Z 轴的误差，单位为度）
    - cumulative_err: [S-1,] 每个时间步的累积旋转误差（仅当 with_cumulative_err=True 时返回，单位为度）
    """

    with torch.no_grad():
        rel_tangle_deg = translation_angle(
            gt_enc[:, :3], pred_pose_enc[:, :3]
        )

        rel_pred_mat = quaternion_to_matrix(pred_pose_enc[..., 3:7])
        rel_gt_mat = quaternion_to_matrix(gt_enc[..., 3:7])


        # get the tracking geodesic distance between prediction and ground-truth
        geodesic_err, error_euler = geodesic_distance_from_two_batches2(
            rel_pred_mat.reshape(-1, 3, 3),
            rel_gt_mat.reshape(-1, 3, 3),
            with_euler_error=True)
        rel_rangle_deg = torch.rad2deg(geodesic_err)
        error_euler = np.rad2deg(error_euler)

        # error_euler: (N, 3) 角度误差 (单位: deg)
        threshold = 5.0
        # 逐列统计 <5° 的比例
        percentages = (error_euler < threshold).mean(axis=0)
        # 转成列表 [x%, y%, z%]
        percentages_list = percentages.tolist()

        avg_rangle_deg = rel_rangle_deg.mean()  # 就是取平均值而已
        error_euler_mean = np.mean(np.abs(error_euler), axis=0)

    return rel_rangle_deg, rel_tangle_deg, avg_rangle_deg, error_euler_mean, percentages_list

def calculate_auc_np(r_error, t_error, max_threshold=30):
    """
    Calculate the Area Under the Curve (AUC) for the given error arrays.

    :param r_error: numpy array representing R error values (Degree).
    :param t_error: numpy array representing T error values (Degree).
    :param max_threshold: maximum threshold value for binning the histogram.
    :return: cumulative sum of normalized histogram of maximum error values.
    """

    # Concatenate the error arrays along a new axis
    error_matrix = np.concatenate((r_error[:, None], t_error[:, None]), axis=1)

    # Compute the maximum error value for each pair
    max_errors = np.max(error_matrix, axis=1)

    # Define histogram bins
    bins = np.arange(max_threshold + 1)

    # Calculate histogram of maximum error values
    histogram, _ = np.histogram(max_errors, bins=bins)

    # Normalize the histogram
    num_pairs = float(len(max_errors))
    normalized_histogram = histogram.astype(float) / num_pairs

    # Compute and return the cumulative sum of the normalized histogram
    return np.mean(np.cumsum(normalized_histogram)), normalized_histogram


def calculate_auc(r_error, t_error, max_threshold=30, return_list=False):
    """
    Calculate the Area Under the Curve (AUC) for the given error arrays using PyTorch.

    :param r_error: torch.Tensor representing R error values (Degree).
    :param t_error: torch.Tensor representing T error values (Degree).
    :param max_threshold: maximum threshold value for binning the histogram.
    :return: cumulative sum of normalized histogram of maximum error values.
    """

    # Concatenate the error tensors along a new axis
    error_matrix = torch.stack((r_error, t_error), dim=1)

    # Compute the maximum error value for each pair
    max_errors, _ = torch.max(error_matrix, dim=1)

    # Define histogram bins
    bins = torch.arange(max_threshold + 1)

    # Calculate histogram of maximum error values
    histogram = torch.histc(
        max_errors, bins=max_threshold + 1, min=0, max=max_threshold
    )

    # Normalize the histogram
    num_pairs = float(max_errors.size(0))
    normalized_histogram = histogram / num_pairs

    if return_list:
        return (
            torch.cumsum(normalized_histogram, dim=0).mean(),
            normalized_histogram,
        )
    # Compute and return the cumulative sum of the normalized histogram
    return torch.cumsum(normalized_histogram, dim=0).mean()


def batched_all_pairs(B, N):
    # B, N = se3.shape[:2]
    i1_, i2_ = torch.combinations(
        torch.arange(N), 2, with_replacement=False
    ).unbind(-1)
    i1, i2 = [
        (i[None] + torch.arange(B)[:, None] * N).reshape(-1) for i in [i1_, i2_]
    ]

    return i1, i2


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


def closed_form_inverse(se3, R=None, T=None):
    """
    Computes the inverse of each 4x4 SE3 matrix in the batch.
    This function assumes PyTorch3D coordinate.


    Args:
    - se3 (Tensor): Nx4x4 tensor of SE3 matrices.

    Returns:
    - Tensor: Nx4x4 tensor of inverted SE3 matrices.
    """
    if R is None:
        R = se3[:, :3, :3]

    if T is None:
        T = se3[:, 3:, :3]

    # NOTE THIS ASSUMES PYTORCH3D CAMERA COORDINATE

    # Compute the transpose of the rotation
    R_transposed = R.transpose(1, 2)

    # Compute the left part of the inverse transformation
    left_bottom = -T.bmm(R_transposed)
    left_combined = torch.cat((R_transposed, left_bottom), dim=1)

    # Keep the right-most column as it is
    right_col = se3[:, :, 3:].detach().clone()
    inverted_matrix = torch.cat((left_combined, right_col), dim=-1)

    return inverted_matrix


def rotation_angle(rot_gt, rot_pred, batch_size=None, eps=1e-15):
    #########
    q_pred = matrix_to_quaternion(rot_pred)
    q_gt = matrix_to_quaternion(rot_gt)
    loss_q = (1 - (q_pred * q_gt).sum(dim=1) ** 2).clamp(min=eps)
    err_q = torch.arccos(1 - 2 * loss_q)

    # loss_q = (1 - (rot_pred * rot_gt).sum(dim=1) ** 2).clamp(min=eps)
    # err_q = torch.arccos(1 - 2 * loss_q)

    rel_rangle_deg = err_q * 180 / np.pi

    if batch_size is not None:
        rel_rangle_deg = rel_rangle_deg.reshape(batch_size, -1)

    return rel_rangle_deg

def rotation_angle2(q_gt, q_pred, batch_size=None, eps=1e-15):
    #########

    loss_q = (1 - (q_pred * q_gt).sum(dim=1) ** 2).clamp(min=eps)
    err_q = torch.arccos(1 - 2 * loss_q)

    rel_rangle_deg = err_q * 180 / np.pi

    if batch_size is not None:
        rel_rangle_deg = rel_rangle_deg.reshape(batch_size, -1)

    return rel_rangle_deg

def translation_angle(tvec_gt, tvec_pred, batch_size=None, ambiguity=True):
    # tvec_gt, tvec_pred (B, 3,)
    rel_tangle_deg = compare_translation_by_angle(tvec_gt, tvec_pred)
    rel_tangle_deg = rel_tangle_deg * 180.0 / np.pi

    if ambiguity:
        rel_tangle_deg = torch.min(rel_tangle_deg, (180 - rel_tangle_deg).abs())

    if batch_size is not None:
        rel_tangle_deg = rel_tangle_deg.reshape(batch_size, -1)

    return rel_tangle_deg


def compare_translation_by_angle(t_gt, t, eps=1e-15, default_err=1e6):
    """Normalize the translation vectors and compute the angle between them."""
    t_norm = torch.norm(t, dim=1, keepdim=True)
    t = t / (t_norm + eps)

    t_gt_norm = torch.norm(t_gt, dim=1, keepdim=True)
    t_gt = t_gt / (t_gt_norm + eps)

    loss_t = torch.clamp_min(1.0 - torch.sum(t * t_gt, dim=1) ** 2, eps)
    err_t = torch.acos(torch.sqrt(1 - loss_t))

    err_t[torch.isnan(err_t) | torch.isinf(err_t)] = default_err
    return err_t
