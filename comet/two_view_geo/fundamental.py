# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from typing import Literal, Optional, Tuple

import torch


import math
from torch.cuda.amp import autocast

from kornia.core.check import KORNIA_CHECK_SHAPE
from kornia.geometry.solvers import solve_cubic

from kornia.geometry.epipolar.fundamental import (
    normalize_points,
    normalize_transformation,
)
from kornia.core import Tensor, concatenate, ones_like, stack, where, zeros

import numpy as np

from .utils import (
    generate_samples,
    sampson_epipolar_distance_batched,
    calculate_residual_indicator,
    normalize_points_masked,
    local_refinement,
    _torch_svd_cast,
    sampson_epipolar_distance_forloop_wrapper,
)


# The code structure learned from https://github.com/kornia/kornia
# Some funtions adapted from https://github.com/kornia/kornia
# The minimal solvers learned from https://github.com/colmap/colmap


def estimate_fundamental(
    points1,
    points2,
    max_ransac_iters=4096,
    max_error=1,
    lo_num=300,
    valid_mask=None,
    squared=True,
    second_refine=True,
    loopresidual=False,
    return_residuals=False,
):
    """
    Given 2D correspondences, 给定一组 2D 对应点，利用 7 点算法加上 LORANSAC（鲁棒随机采样一致性）来估计基础矩阵（Fundamental Matrix）
    this function estimate fundamental matrix by 7pt/8pt algo + LORANSAC.

    points1, points2: Pytorch Tensor, BxNx2

    best_fmat: Bx3x3 估计出来的最佳基础矩阵（形状为 B×3×3）
    """
    max_thres = max_error**2 if squared else max_error #根据是否使用平方误差（squared 参数）调整最大误差阈值

    # points1, points2: BxNx2
    B, N, _ = points1.shape# B(仅仅在这里面）=7 8192
    point_per_sample = 7  # 7p algorithm require 7 pairs 固定每次 RANSAC 采样的点数为 7，因为 7 点算法要求 7 对对应点。

    # randomly sample 7 point set by max_ransac_iters times 。generate_samples 函数生成随机采样的索引，每次采样 7 个点，总共进行 max_ransac_iters 次
    # ransac_idx: torch matirx Nx7
    ransac_idx = generate_samples(N, max_ransac_iters, point_per_sample) #8172  4096 7-》4096，7的数组
    left = points1[:, ransac_idx].view( # 索引法 不要管索引的维数，直接按照索引值对这一维进行 索引即可
        B * max_ransac_iters, point_per_sample, 2
    ) # 利用这些索引，从 points1 和 points2 中抽取对应的 7 点 。并将它们 reshape 成形状 (B * max_ransac_iters=7*28672, 7, 2)。
    right = points2[:, ransac_idx].view(
        B * max_ransac_iters, point_per_sample, 2
    )

    # Note that, we have (B*max_ransac_iters) 7-point sets run_7point 对每个 7 点集合运行 7 点算法，
    # Each 7-point set will lead to 3 potential answers by 7p algorithm (check run_7point for details) 注意 7 点算法可能产生 3 个候选解，所以每组 7 点对应 3 个基础矩阵
    # Therefore the number of 3x3 matrix fmat_ransac is (B*max_ransac_iters*3)
    # We reshape it to B x (max_ransac_iters*3) x 3 x3
    fmat_ransac = run_7point(left, right) # 28672 3 3 3
    fmat_ransac = fmat_ransac.reshape(B, max_ransac_iters, 3, 3, 3).reshape(
        B, max_ransac_iters * 3, 3, 3
    ) # (B, max_ransac_iters * 3, 3, 3) （7 12288 3 3）

    # Not sure why but the computation of sampson errors takes a lot of GPU memory
    # Since it is very fast, users can use a for loop to reduce the peak GPU usage
    # if necessary
    if loopresidual: # 如果 loopresidual 为 True，则采用逐步计算（节省GPU内存），否则批量计算。
        sampson_fn = sampson_epipolar_distance_forloop_wrapper
    else:
        sampson_fn = sampson_epipolar_distance_batched

    residuals = sampson_fn(points1, points2, fmat_ransac, squared=squared) # 7 max_ransac_iters*3=12288 点对数=8192 计算所有候选F矩阵的残差（Sampson距离），计算所有F矩阵对应的所有点对的几何误差。
    if loopresidual:
        torch.cuda.empty_cache()

    # If we know some matches are invalid, 如果提供了 valid_mask，则将对应点对的残差置为很大值，确保这些无效匹配不会被选为内点。
    # we can simply force its corresponding errors as a huge value
    if valid_mask is not None:
        valid_mask_tmp = valid_mask[:, None].expand(-1, residuals.shape[1], -1)
        residuals[~valid_mask_tmp] = 1e6

    # Compute the number of inliers
    # and sort the candidate fmats based on it
    inlier_mask = residuals <= max_thres # 7 12288 8192 通过将残差小于阈值的匹配点视为内点，计算每个候选基础矩阵的内点数量
    inlier_num = inlier_mask.sum(dim=-1) # # 7 12288相当于这些矩阵的符合要求的内点数，就是误差小的点数
    sorted_values, sorted_indices = torch.sort(
        inlier_num, dim=1, descending=True
    ) # 然后对候选解按照内点数进行降序排序。从最好到最差 # 7 12288 # 7 12288

    # Conduct local refinement by 8p algorithm
    # Basically, for a well-conditioned candidate fmat from 7p algorithm
    # we can compute all of its inliers
    # and then feed these inliers to 8p algorithm
    fmat_lo = local_refinement(
        run_8point, points1, points2, inlier_mask, sorted_indices, lo_num=lo_num
    ) # 对前 lo_num=300 个高内点数的候选F矩阵进行局部优化，利用这些内点通过 8 点算法进行局部精炼 7 300 3 3 每个批次300个矩阵
    if loopresidual:
        torch.cuda.empty_cache()
    residuals_lo = sampson_fn(points1, points2, fmat_lo, squared=squared) # 7 300 点对数=8192 计算所有候选F矩阵的残差（Sampson距离），计算所有F矩阵对应的所有点对的几何误差。
    if loopresidual:
        torch.cuda.empty_cache()

    if second_refine: # 如果设置了 second_refine，对 8 点算法的结果再次进行局部精炼，以期获得更优的解
        # We can do this again to the predictd fmats from last run of 8p algorithm
        # Usually it is not necessary but let's put it here
        lo_more = lo_num // 2 # 这次精炼的矩阵数量为150个
        inlier_mask_lo = residuals_lo <= max_thres
        inlier_num_lo = inlier_mask_lo.sum(dim=-1)
        sorted_values_lo, sorted_indices_lo = torch.sort(
            inlier_num_lo, dim=1, descending=True
        )
        fmat_lo_second = local_refinement( #7 150 3 3
            run_8point,
            points1,
            points2,
            inlier_mask_lo,
            sorted_indices_lo,
            lo_num=lo_more,
        )
        if loopresidual:
            torch.cuda.empty_cache()
        residuals_lo_second = sampson_fn(
            points1, points2, fmat_lo_second, squared=squared
        )
        if loopresidual:
            torch.cuda.empty_cache()
        fmat_lo = torch.cat([fmat_lo, fmat_lo_second], dim=1) #7 450 3 3 将两次精炼得到的基础矩阵及残差合并。
        residuals_lo = torch.cat([residuals_lo, residuals_lo_second], dim=1) # 7 450 8192
        lo_num += lo_more

    if valid_mask is not None: # 再一次利用有效掩码调整
        valid_mask_tmp = valid_mask[:, None].expand(
            -1, residuals_lo.shape[1], -1
        )
        residuals_lo[~valid_mask_tmp] = 1e6

    # Get all the predicted fmats
    # choose the one with the highest inlier number and smallest (valid) residual

    all_fmat = torch.cat([fmat_ransac, fmat_lo], dim=1) #7 12288+350=12738 3 3 将7点一开始的所有解和局部精炼后450 的解拼接在一起；
    residuals_all = torch.cat([residuals, residuals_lo], dim=1)# 7 12738 点数8192

    (residual_indicator, inlier_num_all, inlier_mask_all) = (
        calculate_residual_indicator(residuals_all, max_thres, debug=True) # max_thres：误差阈值（经过平方或不平方处理），用于判断一个匹配是否为内点
    ) # 利用 calculate_residual_indicator 计算一个指标（通常结合内点数量和残差），并找到每个批次中指标最高的候选解；对于每个候选基础矩阵，根据残差小于阈值的匹配数量（内点数）进行统计，记为 inlier_num_all；residual_indicator。这个指标通常越高越优。该函数还返回一个内点掩码 inlier_mask_all，标识每个候选解中哪些匹配被认为是内点。

    batch_index = torch.arange(B).unsqueeze(-1).expand(-1, lo_num)# 生成一个一维张量 [0, 1, ..., B-1]-》(B, lo_num=450)

    # Find the index of the best fmat
    best_f_indices = torch.argmax(residual_indicator, dim=1) # 7 返回评价指标最大的候选解的 索引 。
    best_fmat = all_fmat[batch_index[:, 0], best_f_indices] #7 3 3  batch_index[:, 0] 给出每个批次的索引 best_f_indices 给出在每个批次中最佳候选解的索引
    best_inlier_num = inlier_num_all[batch_index[:, 0], best_f_indices]# 对应最佳候选解的内点数量和。
    best_inlier_mask = inlier_mask_all[batch_index[:, 0], best_f_indices] # 内点掩码

    if return_residuals: # 提取对应最佳基础矩阵的残差值 best_residuals，残差越小，说明预测得越准确，
        best_residuals = residuals_all[batch_index[:, 0], best_f_indices]
        return best_fmat, best_inlier_num, best_inlier_mask, best_residuals

    return best_fmat, best_inlier_num, best_inlier_mask


def essential_from_fundamental(
    fmat,
    kmat1,
    kmat2,
    points1=None,
    points2=None,
    focal_length=None,
    principal_point=None,
    max_error=4,
    squared=True,
    compute_residual=False,
):
    """Get Essential matrix from Fundamental and Camera matrices.

    Uses the method from Hartley/Zisserman 9.6 pag 257 (formula 9.12).

    Args:
        F_mat: The fundamental matrix with shape of :math:`(*, 3, 3)`.
        K1: The camera matrix from first camera with shape :math:`(*, 3, 3)`.
        K2: The camera matrix from second camera with shape :math:`(*, 3, 3)`.

    Returns:
        The essential matrix with shape :math:`(*, 3, 3)`.
    """

    with autocast(dtype=torch.float32):
        emat_from_fmat = kmat2.transpose(-2, -1) @ fmat @ kmat1

        if compute_residual:
            principal_point = principal_point.unsqueeze(1)
            focal_length = focal_length.unsqueeze(1)

            points1 = (points1 - principal_point[..., :2]) / focal_length[
                ..., :2
            ]
            points2 = (points2 - principal_point[..., 2:]) / focal_length[
                ..., 2:
            ]

            max_error = max_error / focal_length.mean(dim=-1, keepdim=True)

            max_thres = max_error**2 if squared else max_error

            B, N, _ = points1.shape

            if kmat1 is None:
                raise NotImplementedError

            residuals = sampson_epipolar_distance_batched(
                points1, points2, emat_from_fmat.unsqueeze(1), squared=squared
            )

            inlier_mask = residuals <= max_thres

            inlier_num = inlier_mask.sum(dim=-1).squeeze(1)
            inlier_mask = inlier_mask.squeeze(1)
        else:
            inlier_num = None
            inlier_mask = None

        return emat_from_fmat, inlier_num, inlier_mask


##################################################################
# 8P #
##################################################################


def run_8point(
    points1: Tensor,
    points2: Tensor,
    masks: Optional[Tensor] = None,
    weights: Optional[Tensor] = None,
) -> Tensor:
    r"""Compute the fundamental matrix using the DLT formulation.

    The linear system is solved by using the Weighted Least Squares Solution for the 8 Points algorithm.

    Args:
        points1: A set of points in the first image with a tensor shape :math:`(B, N, 2), N>=8`.
        points2: A set of points in the second image with a tensor shape :math:`(B, N, 2), N>=8`.
        weights: Tensor containing the weights per point correspondence with a shape of :math:`(B, N)`.

    Returns:
        the computed fundamental matrix with shape :math:`(B, 3, 3)`.

    Adapted from https://github.com/kornia/kornia/blob/b0995bdce3b04a11d39e86853bb1de9a2a438ca2/kornia/geometry/epipolar/fundamental.py#L169

    Refer to Hartley and Zisserman, Multiple View Geometry, algorithm 11.1, page 282 for more details

    """
    with autocast(dtype=torch.float32):
        # NOTE: DO NOT use bf16 when related to SVD
        if points1.shape != points2.shape:
            raise AssertionError(points1.shape, points2.shape)
        if points1.shape[1] < 8:
            raise AssertionError(points1.shape)
        if weights is not None:
            if not (
                len(weights.shape) == 2 and weights.shape[1] == points1.shape[1]
            ):
                raise AssertionError(weights.shape)

        if masks is None:
            masks = ones_like(points1[..., 0])

        points1_norm, transform1 = normalize_points_masked(points1, masks=masks)
        points2_norm, transform2 = normalize_points_masked(points2, masks=masks)

        x1, y1 = torch.chunk(points1_norm, dim=-1, chunks=2)  # Bx1xN
        x2, y2 = torch.chunk(points2_norm, dim=-1, chunks=2)  # Bx1xN

        ones = ones_like(x1)

        # build equations system and solve DLT
        # [x * x', x * y', x, y * x', y * y', y, x', y', 1]

        X = torch.cat(
            [x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, ones], dim=-1
        )  # BxNx9

        # if masks is not valid, force the cooresponding rows (points) to all zeros
        if masks is not None:
            X = X * masks.unsqueeze(-1)

        # apply the weights to the linear system
        if weights is None:
            X = X.transpose(-2, -1) @ X
        else:
            w_diag = torch.diag_embed(weights)
            X = X.transpose(-2, -1) @ w_diag @ X

        # compute eigevectors and retrieve the one with the smallest eigenvalue
        _, _, V = _torch_svd_cast(X)
        F_mat = V[..., -1].view(-1, 3, 3)

        # reconstruct and force the matrix to have rank2
        U, S, V = _torch_svd_cast(F_mat)
        rank_mask = torch.tensor(
            [1.0, 1.0, 0.0], device=F_mat.device, dtype=F_mat.dtype
        )

        F_projected = U @ (
            torch.diag_embed(S * rank_mask) @ V.transpose(-2, -1)
        )
        F_est = transform2.transpose(-2, -1) @ (F_projected @ transform1)

        return normalize_transformation(F_est)  # , points1_norm, points2_norm


##################################################################
# 7P #
##################################################################


def run_7point(points1: Tensor, points2: Tensor) -> Tensor:
    # with autocast(dtype=torch.double):
    with autocast(dtype=torch.float32):
        # NOTE: DO NOT use bf16 when related to SVD
        r"""Compute the fundamental matrix using the 7-point algorithm.

        Args:
            points1: A set of points in the first image with a tensor shape :math:`(B, N, 2)`.
            points2: A set of points in the second image with a tensor shape :math:`(B, N, 2)`.

        Returns:
            the computed fundamental matrix with shape :math:`(B, 3*m, 3), Valid values of m are 1, 2 or 3`

        Adapted from:
        https://github.com/kornia/kornia/blob/b0995bdce3b04a11d39e86853bb1de9a2a438ca2/kornia/geometry/epipolar/fundamental.py#L76

        which is based on the following paper:
                Zhengyou Zhang and T. Kanade, Determining the Epipolar Geometry and its
                Uncertainty: A Review, International Journal of Computer Vision, 1998.
                http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.33.4540

        """
        KORNIA_CHECK_SHAPE(points1, ["B", "7", "2"])
        KORNIA_CHECK_SHAPE(points2, ["B", "7", "2"])

        batch_size = points1.shape[0]

        points1_norm, transform1 = normalize_points(points1)
        points2_norm, transform2 = normalize_points(points2)

        x1, y1 = torch.chunk(points1_norm, dim=-1, chunks=2)  # Bx1xN
        x2, y2 = torch.chunk(points2_norm, dim=-1, chunks=2)  # Bx1xN

        ones = ones_like(x1)
        # form a linear system: which represents
        # the equation (x2[i], 1)*F*(x1[i], 1) = 0
        X = concatenate(
            [x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, ones], -1
        )  # BxNx9

        # X * fmat = 0 is singular (7 equations for 9 variables)
        # solving for nullspace of X to get two F
        _, _, v = _torch_svd_cast(X)

        # last two singular vector as a basic of the space
        f1 = v[..., 7].view(-1, 3, 3)
        f2 = v[..., 8].view(-1, 3, 3)

        # lambda*f1 + mu*f2 is an arbitrary fundamental matrix
        # f ~ lambda*f1 + (1 - lambda)*f2
        # det(f) = det(lambda*f1 + (1-lambda)*f2), find lambda
        # form a cubic equation
        # finding the coefficients of cubic polynomial (coeffs)

        coeffs = zeros((batch_size, 4), device=v.device, dtype=v.dtype)

        f1_det = torch.linalg.det(f1)
        f2_det = torch.linalg.det(f2)

        f1_det_invalid = f1_det == 0
        f2_det_invalid = f2_det == 0

        # ignore the samples that failed for det checking
        if f1_det_invalid.any():
            f1[f1_det_invalid] = torch.eye(3).to(f1.device).to(f1.dtype)

        if f2_det_invalid.any():
            f2[f2_det_invalid] = torch.eye(3).to(f2.device).to(f2.dtype)

        coeffs[:, 0] = f1_det
        coeffs[:, 1] = torch.einsum("bii->b", f2 @ torch.inverse(f1)) * f1_det
        coeffs[:, 2] = torch.einsum("bii->b", f1 @ torch.inverse(f2)) * f2_det
        coeffs[:, 3] = f2_det

        # solve the cubic equation, there can be 1 to 3 roots
        roots = solve_cubic(coeffs)

        fmatrix = zeros((batch_size, 3, 3, 3), device=v.device, dtype=v.dtype)
        valid_root_mask = (torch.count_nonzero(roots, dim=1) < 3) | (
            torch.count_nonzero(roots, dim=1) > 1
        )

        _lambda = roots
        _mu = torch.ones_like(_lambda)

        _s = f1[valid_root_mask, 2, 2].unsqueeze(dim=1) * roots[
            valid_root_mask
        ] + f2[valid_root_mask, 2, 2].unsqueeze(dim=1)
        _s_non_zero_mask = ~torch.isclose(
            _s, torch.tensor(0.0, device=v.device, dtype=v.dtype)
        )

        _mu[_s_non_zero_mask] = 1.0 / _s[_s_non_zero_mask]
        _lambda[_s_non_zero_mask] = (
            _lambda[_s_non_zero_mask] * _mu[_s_non_zero_mask]
        )

        f1_expanded = f1.unsqueeze(1).expand(batch_size, 3, 3, 3)
        f2_expanded = f2.unsqueeze(1).expand(batch_size, 3, 3, 3)

        fmatrix[valid_root_mask] = (
            f1_expanded[valid_root_mask]
            * _lambda[valid_root_mask, :, None, None]
            + f2_expanded[valid_root_mask] * _mu[valid_root_mask, :, None, None]
        )

        mat_ind = zeros(3, 3, dtype=torch.bool)
        mat_ind[2, 2] = True
        fmatrix[_s_non_zero_mask, mat_ind] = 1.0
        fmatrix[~_s_non_zero_mask, mat_ind] = 0.0

        trans1_exp = (
            transform1[valid_root_mask]
            .unsqueeze(1)
            .expand(-1, fmatrix.shape[2], -1, -1)
        )
        trans2_exp = (
            transform2[valid_root_mask]
            .unsqueeze(1)
            .expand(-1, fmatrix.shape[2], -1, -1)
        )

        bf16_happy = torch.matmul(
            trans2_exp.transpose(-2, -1),
            torch.matmul(fmatrix[valid_root_mask], trans1_exp),
        )
        fmatrix[valid_root_mask] = bf16_happy.float()

        return normalize_transformation(fmatrix)
