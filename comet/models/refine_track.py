# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

from PIL import Image
import os
from typing import Union, Tuple
from kornia.utils.grid import create_meshgrid
from kornia.geometry.subpix import dsnt

from losses import sequence_loss


def refine_track(
    images,
    fine_fnet,
    fine_tracker,
    coarse_pred, # 与train不一样 这里直接给的是最后一个迭代的结果,
    pradius=15,
    sradius=2,
    compute_score=False,):

    # coarse_pred shape: BxSxNx2,
    # where B is the batch, S is the video/images length, and N is the number of tracks
    # now we are going to extract patches with the center at coarse_pred
    # Please note that the last dimension indicates x and y, and hence has a dim number of 2
    B, S, N, _ = coarse_pred.shape # 1 8 2048
    _, _, _, H, W = images.shape # 1024 1024

    # Given the raidus of a patch, compute the patch size 这里根据 patch 的半径 pradius 计算出 patch 的尺寸 精细跟踪 就是在邻域范围内
    psize = pradius * 2 + 1 # 31

    # Note that we assume the first frame is the query frame 第一帧仍然是查询帧
    # so the 2D locations of the first frame are the query points 和粗跟踪很像 取第一帧的查询点作为查询点
    query_points = coarse_pred[:, 0]

    # Given 2D positions, we can use grid_sample to extract patches 虽然可以用 grid_sample 根据浮点坐标提取 patch，
    # but it takes too much memory. 但 grid_sample 会使用插值操作，内存占用较高且计算量大。
    # Instead, we use the floored track xy to sample patches. 此处采用 “unfold” 操作，这类似于 CNN 中的滑动窗口操作，将整张图像分割成不重叠或重叠的 patch，这个操作在 PyTorch 中经过高度优化

    # For example, if the query point xy is (128.16, 252.78),
    # and the patch size is (31, 31),
    # our goal is to extract the content of a rectangle
    # with left top: (113.16, 237.78)
    # and right bottom: (143.16, 267.78).
    # However, we record the floored left top: (113, 237)
    # and the offset (0.16, 0.78)
    # Then what we need is just unfolding the images like in CNN,
    # picking the content at [(113, 237), (143, 267)].
    # Such operations are highly optimized at pytorch
    # (well if you really want to use interpolation, check the function extract_glimpse() below)

    with torch.no_grad():
        # ====================【修改开始】使用 grid_sample 分支（推理阶段）====================
        # 该分支只提取每个 coarse_pred 附近的 patch，利用 bilinear 插值实现子像素级采样
        # 输入 images shape: (B, S, 3, H, W) -> 重塑为 (B*S, 3, H, W)
        # 使用 grid_sample 方法，只提取粗预测附近的 patch
        # 首先，将图像重塑为 (B*S, 3, H, W)
        content_to_extract = images.reshape(B * S, 3, H, W)  # (B*S, 3, H, W)
        C_in = content_to_extract.shape[1]

        content_to_extract = content_to_extract.unfold(2, psize, 1).unfold(
            # 输出的张量形状会变成类似 (B*S, 3, H_new=1024-31=94, psize, W)
            3, psize, 1
            # 2 表示沿着张量的第 2 维（即图像高度 H）进行操作 步长（stride）为 1，即窗口每次移动 1 个像素 (B*S, 3, H_new=1024-31+1=994,994，31 31)
        )

    # Floor the coarse predictions to get integers and save the fractional/decimal
    track_int = coarse_pred.floor().int() # 对粗略预测的坐标取下整，得到整数坐标（例如 [128.16, 252.78] 变成 [128, 252]），便于后续用于索引操作。
    track_frac = coarse_pred - track_int # 保留原始坐标与整数坐标之间的差值（小数部分），可能在后续精细处理或插值时用到。

    # Note the points represent the center of patches 原始的粗略预测坐标代表 patch 的中心点，而 PyTorch 的 unfold 操作是以 patch 的左上角为索引。
    # now we get the location of the top left corner of patches
    # because the ouput of pytorch unfold are indexed by top left corner
    topleft = track_int - pradius # 得到track对应的 每个 patch 左上角的整数位置
    topleft_BSN = topleft.clone() # 保存一份 topleft_BSN 的拷贝，以便后续使

    # clamp the values so that we will not go out of indexes #使用 clamp 将左上角索引限制在合法范围内，确保不会取到负值或超出图像尺寸。
    # NOTE: (VERY IMPORTANT: This operation ASSUMES H=W).    # 这里假设图像高度 H 与宽度 W 相等；如果不相等，则需要对 x 和 y 分别进行 clamping。
    # You need to seperately clamp x and y if H!=W
    topleft = topleft.clamp(0, H - psize) #1 8 2048 2

    # Reshape from BxSxNx2 -> (B*S)xNx2
    topleft = topleft.reshape(B * S, N, 2)

    # Prepare batches for indexing, shape: (B*S)xN 用来选择哪一张图像
    batch_indices = ( # 生成一个批次索引张量，形状为 (B*S, N)，0000... 1111...  2222...其中每行的值表示当前图像在合并后的批次中的索引
        torch.arange(B * S)[:, None].expand(-1, N).to(content_to_extract.device)
    )

    # Extract image patches based on top left corners
    # extracted_patches: (B*S) x N x C_in x Psize x Psize
    extracted_patches = content_to_extract[
        batch_indices, :, topleft[..., 1], topleft[..., 0]
    ] # batch_indices 为B*S, N  topleft为 B*S, N 2，content_to_extract为B*S，C_in H，W
    # 为什么 extracted_patches不是(B*S, N, C_in, psize, psize)，而是(B*S, N, C_in）
    # 对于每个（B*S）图像，我们根据预先计算好的每个目标的左上角索引，提取出对应的 N 个 patch（新的图像）

    patch_input = extracted_patches.reshape(B * S * N, C_in, psize, psize)

    # Feed patches to fine fent for features  ShallowEncoder
    patch_feat = fine_fnet(patch_input)
    # fine_fnet 对每个 patch(直接想象成图片） 进行特征提取，输出的特征张量 patch_feat 的形状通常为 (B*S*N, latentdim=32, psize, psize)，其中 C_out 为输出通道数

    C_out = patch_feat.shape[1] # 特征提取的维度 怪不得 32

    # Refine the coarse tracks by fine_tracker

    # reshape back to B x S x N x C_out=32 x Psize x Psize
    patch_feat = patch_feat.reshape(B, S, N, C_out, psize, psize)
    patch_feat = rearrange(patch_feat, "b s n c p q -> (b n) s c p q") # (B*N=2048, S=8, C_out=32, psize=31, psize)

    # Prepare for the query points for fine tracker 准备细粒度跟踪器的查询点
    # They are relative to the patch left top corner,
    # instead of the image top left corner now
    # patch_query_points: N x 1 x 2
    # only 1 here because for each patch we only have 1 query point # track_frac # 是先前计算的粗略预测坐标的小数部分（即原始坐标减去 floor 后的部分），代表了目标位置在图像中的精细偏移。
    patch_query_points = track_frac[:, 0] + pradius # 1 2048 2取出第一帧（查询帧）的 frac：每个目标的精细偏移量。+pradius 后，偏移转换到相对于 patch 左上角的坐标系中（因为 patch 左上角距离中心有 pradius 个像素）
    patch_query_points = patch_query_points.reshape(B * N, 2).unsqueeze(1) # (B*N, 1, 2) 相对于patch左上角为原点的位置，但这就是真正的查询点 每个patch 只有一个，因为patch数太大了

    # Feed the PATCH query points and tracks into fine tracker
    # TODO: wait, we can change here right?
    # 迭代次数不一样 iters=6（迭代6次）可能是为了提高预测精度，在推理时进行更多次迭代细化 迭代次数不会影响参数对吧
    fine_pred_track_lists, _, _, query_point_feat,_ = fine_tracker(
        query_points=patch_query_points, fmaps=patch_feat, iters=6, return_feat=True,TRACKorPOSE=False
    )
    # relative the patch top left 最后一次迭代的结果 作为细跟踪的结果
    fine_pred_track = fine_pred_track_lists[-1].clone() #2048=B*N 8 1 2

    # From (relative to the patch top left) to (relative to the image top left)输出的结果从“相对于 patch 左上角”的坐标转换为“相对于原图左上角”的坐标
    for idx in range(len(fine_pred_track_lists)): #对于每一次迭代的结果
        fine_level = rearrange(
            fine_pred_track_lists[idx], "(b n) s u v -> b s n u v", b=B, n=N
        ) # B=1 S=8 N=2048 patch_N'(patch中的查询点个数）=1 2
        fine_level = fine_level.squeeze(-2) # 1 8 2048 2
        fine_level = fine_level + topleft_BSN #这一步将 patch 内部的坐标（原本相对于 patch 的左上角）加上 topleft_BSN（记录了该 patch 在原图中的左上角位置）
        fine_pred_track_lists[idx] = fine_level

    # relative to the image top left
    refined_tracks = fine_pred_track_lists[-1].clone() # 1 8 2048 2只用最后一次迭代的结果
    refined_tracks[:, 0] = query_points  # 并将第一帧（查询帧）的坐标设置为 query_points（原始查询点），确保查询点不被更新

    score = None

    if compute_score:# 不简单呀 通过对查询点处特征的局部相似性热图计算标准差，来量化匹配的可信度或显著性
        score = compute_score_fn(
            query_point_feat,
            patch_feat,
            fine_pred_track,
            sradius,
            psize,
            B,
            N,
            S,
            C_out,
        ) #1 8 2048 bsn

    return refined_tracks, score
    # full_output 是所有的轨迹点（粗+）细， refined_tracks为纯细


def compute_score_fn(
    query_point_feat,
    patch_feat,
    fine_pred_track,
    sradius,
    psize,
    B,
    N,
    S,
    C_out,
):
    """
    Compute the scores, i.e., the standard deviation of the 2D similarity heatmaps,
    given the query point features and reference frame feature maps
    """

    # query_point_feat initial shape: B x N x C_out,
    # query_point_feat indicates the feat at the coorponsing query points
    # Therefore we don't have S dimension here
    query_point_feat = query_point_feat.reshape(B, N, C_out)
    # reshape and expand to B x (S-1) x N x C_out
    query_point_feat = query_point_feat.unsqueeze(1).expand(-1, S - 1, -1, -1)
    # and reshape to (B*(S-1)*N) x C_out
    query_point_feat = query_point_feat.reshape(B * (S - 1) * N, C_out)

    # Radius and size for computing the score
    ssize = sradius * 2 + 1

    # Reshape, you know it, so many reshaping operations
    patch_feat = rearrange(patch_feat, "(b n) s c p q -> b s n c p q", b=B, n=N)

    # Again, we unfold the patches to smaller patches
    # so that we can then focus on smaller patches
    # patch_feat_unfold shape:
    # B x S x N x C_out x (psize - 2*sradius) x (psize - 2*sradius) x ssize x ssize
    # well a bit scary, but actually not
    patch_feat_unfold = patch_feat.unfold(4, ssize, 1).unfold(5, ssize, 1)

    # Do the same stuffs above, i.e., the same as extracting patches
    fine_prediction_floor = fine_pred_track.floor().int()
    fine_level_floor_topleft = fine_prediction_floor - sradius

    # Clamp to ensure the smaller patch is valid
    fine_level_floor_topleft = fine_level_floor_topleft.clamp(0, psize - ssize)
    fine_level_floor_topleft = fine_level_floor_topleft.squeeze(2)

    # Prepare the batch indices and xy locations

    batch_indices_score = torch.arange(B)[:, None, None].expand(
        -1, S, N
    )  # BxSxN
    batch_indices_score = batch_indices_score.reshape(-1).to(
        patch_feat_unfold.device
    )  # B*S*N
    y_indices = fine_level_floor_topleft[..., 0].flatten()  # Flatten H indices
    x_indices = fine_level_floor_topleft[..., 1].flatten()  # Flatten W indices

    reference_frame_feat = patch_feat_unfold.reshape(
        B * S * N, C_out, psize - sradius * 2, psize - sradius * 2, ssize, ssize
    )

    # Note again, according to pytorch convention
    # x_indices cooresponds to [..., 1] and y_indices cooresponds to [..., 0]
    reference_frame_feat = reference_frame_feat[
        batch_indices_score, :, x_indices, y_indices
    ]
    reference_frame_feat = reference_frame_feat.reshape(
        B, S, N, C_out, ssize, ssize
    )
    # pick the frames other than the first one, so we have S-1 frames here
    reference_frame_feat = reference_frame_feat[:, 1:].reshape(
        B * (S - 1) * N, C_out, ssize * ssize
    )

    # Compute similarity
    sim_matrix = torch.einsum(
        "mc,mcr->mr", query_point_feat, reference_frame_feat
    )
    softmax_temp = 1.0 / C_out**0.5
    heatmap = torch.softmax(softmax_temp * sim_matrix, dim=1)
    # 2D heatmaps
    heatmap = heatmap.reshape(
        B * (S - 1) * N, ssize, ssize
    )  # * x ssize x ssize

    coords_normalized = dsnt.spatial_expectation2d(heatmap[None], True)[0]
    grid_normalized = create_meshgrid(
        ssize, ssize, normalized_coordinates=True, device=heatmap.device
    ).reshape(1, -1, 2)

    var = (
        torch.sum(
            grid_normalized**2 * heatmap.view(-1, ssize * ssize, 1), dim=1
        )
        - coords_normalized**2
    )
    std = torch.sum(
        torch.sqrt(torch.clamp(var, min=1e-10)), -1
    )  # clamp needed for numerical stability

    score = std.reshape(B, S - 1, N)
    # set score as 1 for the query frame
    score = torch.cat([torch.ones_like(score[:, 0:1]), score], dim=1)

    return score


def extract_glimpse(
    tensor: torch.Tensor,
    size: Tuple[int, int],
    offsets,
    mode="bilinear",
    padding_mode="zeros",
    debug=False,
    orib=None,
):
    B, C, W, H = tensor.shape

    h, w = size
    xs = (
        torch.arange(0, w, dtype=tensor.dtype, device=tensor.device)
        - (w - 1) / 2.0
    )
    ys = (
        torch.arange(0, h, dtype=tensor.dtype, device=tensor.device)
        - (h - 1) / 2.0
    )

    vy, vx = torch.meshgrid(ys, xs)
    grid = torch.stack([vx, vy], dim=-1)  # h, w, 2
    grid = grid[None]

    B, N, _ = offsets.shape

    offsets = offsets.reshape((B * N), 1, 1, 2)
    offsets_grid = offsets + grid

    # normalised grid  to [-1, 1]
    offsets_grid = (
        offsets_grid - offsets_grid.new_tensor([W / 2, H / 2])
    ) / offsets_grid.new_tensor([W / 2, H / 2])

    # BxCxHxW -> Bx1xCxHxW
    tensor = tensor[:, None]

    # Bx1xCxHxW -> BxNxCxHxW
    tensor = tensor.expand(-1, N, -1, -1, -1)

    # BxNxCxHxW -> (B*N)xCxHxW
    tensor = tensor.reshape((B * N), C, W, H)

    sampled = torch.nn.functional.grid_sample(
        tensor,
        offsets_grid,
        mode=mode,
        align_corners=False,
        padding_mode=padding_mode,
    )

    # NOTE: I am not sure it should be h, w or w, h here
    # but okay for sqaures
    sampled = sampled.reshape(B, N, C, h, w)

    return sampled


def extract_patches(images, coarse_pred, fmaps=None, cfg=None, is_train=False):
    B, S, N, _ = coarse_pred.shape
    _, _, H, W = images.shape[-4:]
    psize = cfg['patch_size']
    pradius = psize // 2

    # ==================== 统一使用 grid_sample 方法 ====================
    # 将图像重塑为 (B*S, 3, H, W)
    content_to_extract = images.view(-1, 3, H, W)

    # 生成基础采样网格（向量化）
    grid_lin = torch.linspace(-pradius, pradius, psize, device=images.device)
    base_y, base_x = torch.meshgrid(grid_lin, grid_lin, indexing='xy')
    base_grid = torch.stack([base_x, base_y], dim=-1)  # (psize, psize, 2)

    # 批量生成所有采样点（B*S*N个中心点）
    centers = coarse_pred.view(-1, N, 2)  # (B*S, N, 2)

    # 构造归一化网格（向量化处理）
    grid = centers.view(-1, 1, 1, 2) + base_grid.view(1, psize, psize, 2)  # (B*S*N, psize, psize, 2)
    grid_norm = torch.empty_like(grid)
    grid_norm[..., 0] = 2.0 * grid[..., 0] / (W - 1) - 1  # x坐标归一化
    grid_norm[..., 1] = 2.0 * grid[..., 1] / (H - 1) - 1  # y坐标归一化

    # 批量采样（内存优化关键）
    content_expanded = content_to_extract[:, None].expand(-1, N, -1, -1, -1)  # (B*S, N, 3, H, W)
    extracted_patches = F.grid_sample(
        content_expanded.reshape(-1, 3, H, W),
        grid_norm.view(-1, psize, psize, 2),
        mode='bilinear',
        align_corners=True
    ).view(B * S, N, 3, psize, psize)

    # ==================== 特征融合优化 ====================
    if cfg.get("share_coarse_features", False) and (fmaps is not None):
        _, C, H8, W8 = fmaps.shape[-4:]
        scale = H / H8  # 修正尺度计算

        # 动态计算特征图patch尺寸
        psize_fmap = max(1, int(round(psize / scale)))
        if psize_fmap % 2 == 0:  # 确保为奇数以获得对称采样
            psize_fmap += 1

        # 生成特征采样网格（复用图像采样逻辑）
        f_base = torch.linspace(-psize_fmap // 2, psize_fmap // 2, psize_fmap, device=fmaps.device)
        f_base_y, f_base_x = torch.meshgrid(f_base, f_base, indexing='xy')
        f_base_grid = torch.stack([f_base_x, f_base_y], dim=-1)

        # 特征图坐标计算（向量化）
        f_centers = (coarse_pred / scale).view(-1, N, 2)
        f_grid = f_centers.view(-1, 1, 1, 2) + f_base_grid.view(1, psize_fmap, psize_fmap, 2)

        # 特征图归一化
        f_grid_norm = torch.empty_like(f_grid)
        f_grid_norm[..., 0] = 2.0 * f_grid[..., 0] / (W8 - 1) - 1
        f_grid_norm[..., 1] = 2.0 * f_grid[..., 1] / (H8 - 1) - 1

        # 批量特征采样
        fmaps_reshaped = fmaps.view(-1, C, H8, W8)
        f_expanded = fmaps_reshaped[:, None].expand(-1, N, -1, -1, -1).reshape(-1, C, H8, W8)
        feature_patches = F.grid_sample(
            f_expanded,
            f_grid_norm.view(-1, psize_fmap, psize_fmap, 2),
            mode='bilinear',
            align_corners=True
        )

        # 动态上采样替代显式存储
        fused_patches = torch.cat([
            extracted_patches.view(-1, 3, psize, psize),
            F.interpolate(feature_patches, (psize, psize), mode='bilinear', align_corners=True)
        ], dim=1)
    else:
        fused_patches = extracted_patches.view(-1, 3, psize, psize)

    return fused_patches
