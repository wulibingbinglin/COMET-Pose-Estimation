# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

# from cotracker.models.core.model_utils import reduce_masked_mean

EPS = 1e-9


def reduce_masked_mean(input, mask, dim=None, keepdim=False):
    r"""Masked mean
    它是一个带掩码（mask）的均值计算函数。在计算均值时，会考虑 mask 进行加权计算，忽略无效值（mask=0 的部分）

    `reduce_masked_mean(x, mask)` computes the mean of a tensor :attr:`input`
    over a mask :attr:`mask`, returning

    .. math::
        \text{output} =
        \frac
        {\sum_{i=1}^N \text{input}_i \cdot \text{mask}_i}
        {\epsilon + \sum_{i=1}^N \text{mask}_i}

    where :math:`N` is the number of elements in :attr:`input` and
    :attr:`mask`, and :math:`\epsilon` is a small constant to avoid
    division by zero.

    `reduced_masked_mean(x, mask, dim)` computes the mean of a tensor
    :attr:`input` over a mask :attr:`mask` along a dimension :attr:`dim`.
    Optionally, the dimension can be kept in the output by setting
    :attr:`keepdim` to `True`. Tensor :attr:`mask` must be broadcastable to
    the same dimension as :attr:`input`.

    The interface is similar to `torch.mean()`.

    Args:
        inout (Tensor): input tensor.
        mask (Tensor): mask.
        dim (int, optional): Dimension to sum over. Defaults to None.
        keepdim (bool, optional): Keep the summed dimension. Defaults to False.

    Returns:
        Tensor: mean tensor.
    """

    mask = mask.expand_as(input) # mask 可能比 input 维度少，所以扩展（expand_as）以匹配 input

    prod = input * mask # 只有 mask=1 的地方会被保留，mask=0 的地方会变成 0。

    if dim is None:
        numer = torch.sum(prod) # 对 input 进行加权求和（只求 mask=1 处的值）
        denom = torch.sum(mask) # 计算 mask=1 的元素个数（避免不必要的 0 影响）
    else: # 如果 dim 不是 None：只在指定 dim 维度上求均值
        numer = torch.sum(prod, dim=dim, keepdim=keepdim)
        denom = torch.sum(mask, dim=dim, keepdim=keepdim)

    mean = numer / (EPS + denom) # EPS（一个极小值）用于避免除零错误。
    return mean # 如果 denom=0，不会报错，而是会返回 0（因为 EPS 很小）。

#
# def balanced_ce_loss(pred, gt, valid=None):
#     # pred and gt are the same shape
#     for a, b in zip(pred.size(), gt.size()):
#         # print(p)
#         assert a == b  # some shape mismatch!
#
#     if valid is not None:
#         for a, b in zip(pred.size(), valid.size()):
#             assert a == b  # some shape mismatch!
#     else:
#         valid = torch.ones_like(gt)
#
#     pos = (gt > 0.95).to(gt.dtype)
#     neg = (gt < 0.05).to(gt.dtype)
#
#     label = pos * 2.0 - 1.0
#     a = -label * pred
#     b = F.relu(a)
#     loss = b + torch.log(torch.exp(-b) + torch.exp(a - b))
#
#     pos_loss = reduce_masked_mean(loss, pos * valid)
#     neg_loss = reduce_masked_mean(loss, neg * valid)
#
#     balanced_loss = pos_loss + neg_loss
#
#     return balanced_loss, loss

def balanced_ce_loss(pred, gt, valid=None):
    """
    计算平衡的交叉熵损失
    Args:
        pred: 预测值 应为 [B, S, N] 或 [B*S, N]
        gt: 真实值 应为 [B, S, N] 或 [B*S, N]
        valid: 有效性掩码 应为 [B, S, N] 或 [B*S, N]，或 None
    """
    # 打印输入张量的基本信息
    print("\n=== Balanced CE Loss Debug Info ===")
    print(f"Pred tensor: shape={pred.shape}, dtype={pred.dtype}, device={pred.device}")
    print(f"GT tensor: shape={gt.shape}, dtype={gt.dtype}, device={gt.device}")

    # 检查 pred 和 gt 的形状匹配
    if pred.shape != gt.shape:
        raise ValueError(
            f"\nShape mismatch between pred and gt:"
            f"\npred shape: {pred.shape}"
            f"\ngt shape: {gt.shape}"
            f"\nDifference in dimensions: {[a - b for a, b in zip(pred.shape, gt.shape)]}"
        )

    # 检查数值范围
    print("\nValue ranges:")
    print(f"pred range: [{pred.min().item():.3f}, {pred.max().item():.3f}]")
    print(f"gt range: [{gt.min().item():.3f}, {gt.max().item():.3f}]")

    # 检查并处理 valid 掩码
    if valid is not None:
        print(f"Valid mask: shape={valid.shape}, dtype={valid.dtype}, device={valid.device}")
        if valid.shape != pred.shape:
            raise ValueError(
                f"\nShape mismatch between pred and valid:"
                f"\npred shape: {pred.shape}"
                f"\nvalid shape: {valid.shape}"
            )
    else:
        print("Valid mask is None, creating ones_like(gt)")
        valid = torch.ones_like(gt)

    # 计算正负样本比例
    pos = (gt > 0.95).to(gt.dtype)
    neg = (gt < 0.05).to(gt.dtype)

    pos_count = pos.sum().item()
    neg_count = neg.sum().item()
    total_count = gt.numel()

    print("\nClass distribution:")
    print(f"Positive samples: {pos_count} ({pos_count / total_count * 100:.2f}%)")
    print(f"Negative samples: {neg_count} ({neg_count / total_count * 100:.2f}%)")
    print(
        f"Middle range samples: {total_count - pos_count - neg_count} ({(total_count - pos_count - neg_count) / total_count * 100:.2f}%)")

    # 损失计算
    label = pos * 2.0 - 1.0
    a = -label * pred
    b = F.relu(a)
    loss = b + torch.log(torch.exp(-b) + torch.exp(a - b))

    # 检查损失值是否有异常
    if torch.isnan(loss).any():
        print("\nWARNING: NaN values detected in loss!")
        nan_count = torch.isnan(loss).sum().item()
        print(f"Number of NaN values: {nan_count}")

    if torch.isinf(loss).any():
        print("\nWARNING: Inf values detected in loss!")
        inf_count = torch.isinf(loss).sum().item()
        print(f"Number of Inf values: {inf_count}")

    # 计算正负样本的损失
    pos_loss = reduce_masked_mean(loss, pos * valid)
    neg_loss = reduce_masked_mean(loss, neg * valid)

    print("\nLoss values:")
    print(f"Positive loss: {pos_loss.item():.4f}")
    print(f"Negative loss: {neg_loss.item():.4f}")

    balanced_loss = pos_loss + neg_loss
    print(f"Balanced loss: {balanced_loss.item():.4f}")
    print("=" * 40)

    return balanced_loss, loss


def huber_loss(x, y, delta=1.0):
    """Calculate element-wise Huber loss between x and y"""
    diff = x - y
    abs_diff = diff.abs()
    flag = (abs_diff <= delta).to(diff.dtype)
    return flag * 0.5 * diff**2 + (1 - flag) * delta * (abs_diff - 0.5 * delta)


def sequence_loss(
    flow_preds,   # 预测的光流序列（列表，每个元素是 BxSxNx2）
    flow_gt,      # 真实的光流（BxSxNx2）
    vis,          # 目标的可见性 (BxSxN)，1 代表可见，0 代表遮挡
    valids,       # 目标的有效性 (BxSxN)，1 代表有效，0 代表无效
    gamma=0.8,    # 预测的衰减因子（靠近 GT 的预测贡献更大） 权重衰减因子，对较晚的预测赋予更小权重。
    vis_aware=False, # 是否考虑可见性
    huber=False,  # 是否使用 Huber Loss ,如果 True，使用 Huber Loss，否则使用 L1 Loss。
    delta=10,     # Huber Loss 的阈值
    vis_aware_w=0.1, # 可见性权重（调整不可见点的损失贡献）
    ignore_first=False, # 是否忽略第一帧, 如果 True，损失计算时忽略第一帧（可能是因为第一帧被用作初始化）
    max_thres=-1  # 设定最大误差阈值（大于该阈值的误差点会被过滤掉）
):
    """Loss function defined over sequence of flow predictions"""
    B, S, N, D = flow_gt.shape
    assert D == 2
    B, S1, N = vis.shape
    B, S2, N = valids.shape
    assert S == S1
    assert S == S2
    n_predictions = len(flow_preds)
    flow_loss = 0.0

    if ignore_first: # 如果 ignore_first=True，丢弃第一帧的数据。 因为第一帧本就是正确的.第一帧的 GT 可能是初始化值，而非真实流动。
        flow_gt = flow_gt[:, 1:]
        vis = vis[:, 1:]
        valids = valids[:, 1:]

    for i in range(n_predictions): # 对于每次迭代的结果
        i_weight = gamma ** (n_predictions - i - 1) # 让后期的预测贡献更小（指数衰减）
        flow_pred = flow_preds[i]

        if ignore_first:
            flow_pred = flow_pred[:, 1:]

        if huber: # 使用 L1 Loss (|flow_pred - flow_gt|) 或 Huber Loss
            i_loss = huber_loss(flow_pred, flow_gt, delta)  # B, S, N, 2
        else:
            i_loss = (flow_pred - flow_gt).abs()  # B, S, N, 2

        i_loss = torch.nan_to_num(i_loss, nan=0.0, posinf=0.0, neginf=0.0)
        # 处理 NaN 和 inf，避免异常值影响训练。

        if max_thres > 0: # 如果设定了 max_thres，就筛掉误差过大的点（i_loss < max_thres）
            valids = torch.logical_and(valids, (i_loss < max_thres).any(dim=-1))

        i_loss = torch.mean(i_loss, dim=3)  # B, S, N # 计算 (dx, dy) 误差的均值：

        if vis_aware:
            if vis_aware_w==0: # vis_aware_w=0 时，完全丢弃不可见点（valids = valids & vis
                valids = torch.logical_and(valids, vis)
                # i_loss = reduce_masked_mean(i_loss, vis, dim=3)
            else:
                i_loss = i_loss * (vis.to(i_loss.dtype) + vis_aware_w)


        flow_loss += i_weight * reduce_masked_mean(i_loss, valids)
        # 计算只在 valids=1 处的均值（避免背景区域影响） 乘上 i_weight，对后期预测进行权重衰减。

    # clip_trackL
    flow_loss = flow_loss / n_predictions

    return flow_loss
