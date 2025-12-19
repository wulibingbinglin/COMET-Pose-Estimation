# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn.functional as F
from torch import nn

from hydra.utils import instantiate
from comet.models.refine_track import refine_track


class TrackerPredictor(nn.Module): # TrackerPredictor 是一个神经网络模块类，用于特征点跟踪。
    def __init__(
        self,
        COARSE, # 字典，用于构建 粗略预测器（特征提取网络和跟踪器）
        FINE, # 字典，用于构建 精细预测器
        stride=4, # 默认步幅，控制下采样率。
        corr_levels=5, # 默认 5 相关性级别，可能用于构建金字塔特征或多层相关匹配。
        corr_radius=4, # 相关匹配半径，定义局部搜索窗口的大小
        latent_dim=128, # 隐藏层维度，通常指网络的特征通道数
        cfg=None,
        **extra_args # 接受额外的关键字参数，允许灵活扩展。
    ):
        super(TrackerPredictor, self).__init__() # 调用父类 nn.Module 的 __init__() 方法，确保 TrackerPredictor 能够正确继承父类功能。
        """
        COARSE and FINE are the dicts to construct the modules COARSE 和 FINE 是两个字典，用于构造不同的模块。

        Both coarse_predictor and fine_predictor are constructed as a BaseTrackerPredictor,
        check track_modules/base_track_predictor.py
        
        Both coarse_fnet and fine_fnet are constructed as a 2D CNN network
        check track_modules/blocks.py for BasicEncoder and ShallowEncoder 
        """

        self.cfg = cfg

        # coarse predictor
        self.coarse_down_ratio = COARSE.down_ratio # 粗略跟踪的下采样率为2 down_ratio 表示 下采样率，通常用于特征提取网络的下采样过程
        self.coarse_fnet = instantiate(
            COARSE.FEATURENET, _recursive_=False, stride=COARSE.stride, cfg=cfg
        ) # instantiate：实例化一个模块，FEATURENET 是一个 CNN 特征提取网络（如 BasicEncoder
        self.coarse_predictor = instantiate(
            COARSE.PREDICTOR, _recursive_=False, stride=COARSE.stride, cfg=cfg
        ) # 实例化粗略预测器，用于初步预测特征点的匹配或跟踪

        # fine predictor, forced to use stride = 1
        self.fine_fnet = instantiate(
            FINE.FEATURENET, _recursive_=False, stride=1, cfg=cfg
        ) # stride=1：保留原始分辨率，就那么大。
        self.fine_predictor = instantiate(
            FINE.PREDICTOR, _recursive_=False, stride=1, cfg=cfg
        )

    # def forward(
    #     self,
    #     images,
    #     query_points,
    #     fmaps=None,
    #     coarse_iters=6,
    #     inference=True,
    #     fine_tracking=True,
    # ):
    #     """
    #     Args:
    #         images (torch.Tensor): Images as RGB, in the range of [0, 1], with a shape of B x S x 3 x H x W.
    #         query_points (torch.Tensor): 2D xy of query points, relative to top left, with a shape of B x N x 2.
    #         fmaps (torch.Tensor, optional): Precomputed feature maps. Defaults to None.
    #         coarse_iters (int, optional): Number of iterations for coarse prediction. Defaults to 6.
    #         inference (bool, optional): Whether to perform inference. Defaults to True.
    #         fine_tracking (bool, optional): Whether to perform fine tracking. Defaults to True. 如果为 True，则在粗略预测基础上进一步细化轨迹预测
    #
    #     Returns:
    #         tuple: A tuple containing fine_pred_track, coarse_pred_track, pred_vis, and pred_score.
    #     """
    #
    #     if fmaps is None:# 通过 self.process_images_to_fmaps(images) 将输入图像转换为特征图：
    #         fmaps = self.process_images_to_fmaps(images)
    #
    #     if inference:
    #         torch.cuda.empty_cache()
    #
    #     # Coarse prediction
    #     coarse_pred_track_lists, pred_vis = self.coarse_predictor(
    #         query_points=query_points,
    #         fmaps=fmaps,
    #         iters=coarse_iters,
    #         down_ratio=self.coarse_down_ratio,
    #     )
    #     coarse_pred_track = coarse_pred_track_lists[-1] # 取列表中的迭代的最后一次作为最终的粗略预测轨迹 coarse_pred_track 1 8 2048 2
    #
    #     if inference:
    #         torch.cuda.empty_cache()
    #
    #     if fine_tracking:
    #         # Refine the coarse prediction
    #         fine_pred_track, pred_score = refine_track(
    #             images,
    #             self.fine_fnet,
    #             self.fine_predictor,
    #             coarse_pred_track,
    #             compute_score=True,
    #             cfg=self.cfg,
    #         )
    #
    #         if inference:
    #             torch.cuda.empty_cache()
    #     else:
    #         fine_pred_track = coarse_pred_track
    #         pred_score = torch.ones_like(pred_vis) # 得分为1
    #
    #     return fine_pred_track, coarse_pred_track, pred_vis, pred_score # 第一参考帧的得分永远为一

    def process_images_to_fmaps(self, images, training=False):
        """
        This function processes images for inference.

        Args:
            images (np.array): The images to be processed.

        Returns:
            np.array: The processed images.
        """
        batch_num, frame_num, image_dim, height, width = images.shape #(1 8 3 1024 1024)
        if not training:
            assert (
                batch_num == 1
            ), "now we only support processing one scene during inference"
        reshaped_image = images.reshape(
            batch_num * frame_num, image_dim, height, width
        ) # 何不直接传入已经变形过的图像
        if self.coarse_down_ratio > 1:
            # whether or not scale down the input images to save memory # 是否缩小输入图像以节省显存
            fmaps = self.coarse_fnet(
                F.interpolate( # F.interpolate() 用于对 reshaped_image 进行下采样（bilinear 插值）
                    reshaped_image,
                    scale_factor=1 / self.coarse_down_ratio, # scale_factor=1 / self.coarse_down_ratio 代表缩小的比例
                    mode="bilinear",
                    align_corners=True,
                )
            )
        else:
            fmaps = self.coarse_fnet(reshaped_image)
        fmaps = fmaps.reshape(
            batch_num, frame_num, -1, fmaps.shape[-2], fmaps.shape[-1]
        ) # 最后恢复维度即可 -1位置是提取的特征大小 1 8 128 128 128

        return fmaps
