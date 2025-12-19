# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from einops import rearrange, repeat

from utils import get_1d_sincos_pos_embed_from_grid,sample_features4d, get_2d_embedding, get_2d_sincos_pos_embed
from .blocks import EfficientUpdateFormer, CorrBlock, EfficientCorrBlock


class BaseTrackerPredictor(nn.Module):
    def __init__(
        self,
        stride=4,# stride = 粗跟踪的stride 以下都是默认值，除了cfg
        corr_levels=5,
        corr_radius=4,
        latent_dim=128,
        hidden_size=384,
        use_spaceatt=True,
        depth=6,
        fine=False, #base对应的就是粗略呀，所以没有fine情有可原
        cfg=None,
    ):
        # 在细跟踪的时候cfg  stride=1
        #                 depth: 4 =空间深度 = 时间深度
        #                 corr_levels: 3
        #                 corr_radius: 3
        #                 latent_dim: 32
        #                 hidden_size: 256
        #                 fine: True
        #                 use_spaceatt: False 不使用空间

        super(BaseTrackerPredictor, self).__init__()
        """
        The base template to create a track predictor
        
        Modified from https://github.com/facebookresearch/co-tracker/
        就是cotracker2类
        """

        self.cfg = cfg

        self.stride = stride  # 4 下采样的尺度
        self.latent_dim = latent_dim # 128，输入的维度，就是特征向量的维度
        self.corr_levels = corr_levels # 表示金字塔的层数
        self.corr_radius = corr_radius # 用于定义计算相关性时窗口的半径
        self.hidden_size = hidden_size # 掩藏层
        self.fine = fine

        self.flows_emb_dim = latent_dim // 2 #64
        self.transformer_dim = ( # 664
            self.corr_levels * (self.corr_radius * 2 + 1) ** 2
            + self.latent_dim * 2
        ) # 前半部分是cotracker中的fcorrs，然后加上两个latent_dim维的两个向量

        self.efficient_corr = cfg.MODEL.TRACK.efficient_corr # efficient_corr 对corr 控制是否启用高效的相关性计算，通常用于优化计算过程，减少计算量

        if self.fine:
            # TODO this is the old dummy code, will remove this when we train next model
            self.transformer_dim += 4 if self.transformer_dim % 2 == 0 else 5 # 如果 fine 为 True，则增加 4 或 5 的维度，保证双数且 更精细地处理特征维度。
        else:
            self.transformer_dim += (4 - self.transformer_dim % 4) % 4 # 保证 self.transformer_dim 是 4 的倍数。664

        space_depth = depth if use_spaceatt else 0 # space_depth 用于处理空间特征
        time_depth = depth # 一定有time 同cotracker

        self.updateformer = EfficientUpdateFormer( # cotracker中的内容
            space_depth=space_depth,
            time_depth=time_depth,
            input_dim=self.transformer_dim, # 每个轨迹点的维度，就是tokens，连接在一起的维度
            hidden_size=self.hidden_size,
            output_dim=self.latent_dim + 2,# 130
            mlp_ratio=4.0,
            add_space_attn=use_spaceatt, #没有numvirtual
        ) # 就是用来更新的

        self.norm = nn.GroupNorm(1, self.latent_dim)

        # A linear layer to update track feats at each iteration
        self.ffeat_updater = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim), nn.GELU()
        )

        if self.cfg.track_conf: #  预测特征点的置信度 (conf_predictor)
            self.conf_predictor = nn.Sequential(nn.Linear(self.latent_dim, 1)) # 和 vis_predictor 类似，输入 latent_dim，输出 1 维的分数


        if not self.fine: # 如果 fine 为 False，则会定义一个简单的线性层 nn.Linear(self.latent_dim, 1)，用于预测某种特征，例如某个视觉预测值（例如置信度、概率等）。
            self.vis_predictor = nn.Sequential(nn.Linear(self.latent_dim, 1))

    def forward(
        self, query_points, fmaps=None, iters=4, return_feat=False, down_ratio=1
    , is_train=False,track_feats=None,TRACKorPOSE=True,ind=0):
        """
        query_points: B x N x 2, the number of batches, tracks, and xy
        fmaps: B x S x C x HH x WW, the number of batches, frames, and feature dimension.
                note HH and WW is the size of feature maps instead of original images
        """
        if TRACKorPOSE:
            B, S, N, D = query_points.shape # 1 2048 2
        else:
            B, N, D = query_points.shape # 1 2048 2
        B, S, C, HH, WW = fmaps.shape # 1 8 128 128 128

        assert D == 2

        # Scale the input query_points because we may downsample the images
        # by down_ratio or self.stride
        # e.g., if a 3x1024x1024 image is processed to a 128x256x256 feature map
        # its query_points should be query_points/4
        if down_ratio > 1: # 当输入图像经过下采样（down_ratio 或网络的 stride）后，特征图的尺寸会减小。因此需要对原始查询点坐标进行缩放，使其与特征图的分辨率对应。
            query_points = query_points / float(down_ratio)  # 1 2048 2
            query_points = query_points / float(self.stride) # 1 2048 2

        # Init with coords as the query points 初始时，每个帧中的查询点位置都和参考帧的查询点一致，后续可能会根据相关性进行调整
        # It means the search will start from the position of query points at the reference frames 就和cotracker一开始按照第一帧进行初始化一样
        # B, N, D = query_points.shape # 1 2048 2  有它才有它
        # coords = query_points.clone().reshape(B, 1, N, 2).repeat(1, S, 1, 1) # B N 2-> BSN2 1 8 2048 2
        if TRACKorPOSE:
            coords = query_points.clone()
        else:
            coords = query_points.clone().reshape(B, 1, N, 2).repeat(1, S, 1, 1) # B N 2-> BSN2 1 8 2048 2

        # # Sample/extract the features of the query points 轨迹特征 in the query frame
        ##############################用第一帧#############################
        query_track_feat = sample_features4d(fmaps[:, 0], coords[:, 0]) # 1 2048 C=128从第一帧（索引为 0）的特征图中提取出查询点位置的特征。
        # #
        # # # init track feats by query feats
        track_feats = query_track_feat.unsqueeze(1).repeat( # 利用参考帧提取的特征作为所有帧的初始轨迹特征。
            1, S, 1, 1
        )  # B, S, N, C
        # # # back up the init coords

        coords_backup = coords.clone() # 保存一份 coords 的拷贝，后续可能会用到原始的查询点坐标

        # Construct the correlation block
        if self.efficient_corr: # 根据是否启用高效相关性计算（efficient_corr 标志），先采样 再求相关性 好像不启用
            fcorr_fn = EfficientCorrBlock(
                fmaps, num_levels=self.corr_levels, radius=self.corr_radius
            ) # (B, N, S, 层数*（LRR**2）)
        else:
            fcorr_fn = CorrBlock( # 先求相关性 再采样
                fmaps, num_levels=self.corr_levels, radius=self.corr_radius
            ) # 这就是建立一个类 这个类和cotracker 一模一样 不同是值不同 出来是 B N S (2*corr_radius+1)**2*corr_levels

        coord_preds = []# 和cotracker一样

        # Iterative Refinement
        for itr in range(iters):
            # Detach the gradients from the last iteration
            # (in my experience, not very important for performance)
            coords = coords.detach()

            # Compute the correlation (check the implementation of CorrBlock)
            if self.efficient_corr: # no
                fcorrs = fcorr_fn.sample(coords, track_feats)
            else: # 这个是训练时候用的
                fcorr_fn.corr(track_feats)
                fcorrs = fcorr_fn.sample(coords)  # B, S, N, LRR = 1 8 2048 405=91*5

            corrdim = fcorrs.shape[3] #405

            fcorrs_ = fcorrs.permute(0, 2, 1, 3).reshape(B * N, S, corrdim) #  (B, S, N, corrdim) -》2048 8 405

            # Movement of current coords relative to query points
            flows = (
                (coords - coords[:, 0:1]) # 计算当前坐标与参考帧（通常为第一帧）之间的差值，得到每个点在每一帧相对于第一帧的位移；
                .permute(0, 2, 1, 3) # 1 2048 8 2
                .reshape(B * N, S, 2) # 2048 8 2
            )

            flows_emb = get_2d_embedding(
                flows, self.flows_emb_dim, cat_coords=False
            ) # 2048 8 128 # N S E # (B * N, S, C*3\C*3+3)

            # (In my trials, it is also okay to just add the flows_emb instead of concat) 实验中发现直接相加也可，但这里选择拼接，以保留更多信息。
            flows_emb = torch.cat([flows_emb, flows], dim=-1)# 2048 8 130=128+2

            track_feats_ = track_feats.permute(0, 2, 1, 3).reshape(
                B * N, S, self.latent_dim
            ) ## (B, S, N, latent_dim) ->(B * N, S, latent_dim)

            # Concatenate them as the input for the transformers 同cotracker# Get the flow embeddings
            transformer_input = torch.cat(
                [flows_emb, fcorrs_, track_feats_], dim=2
            ) # 2048 8 663=130+405+128 fine:213

            if transformer_input.shape[2] < self.transformer_dim: # 对 Transformer 输入做维度补齐
                # pad the features to match the dimension
                pad_dim = self.transformer_dim - transformer_input.shape[2]
                pad = torch.zeros_like(flows_emb[..., 0:pad_dim]) # 利用 flows_emb 的部分维度创建一个形状相同的全零张量 pad。
                transformer_input = torch.cat([transformer_input, pad], dim=2).to(query_points.device)

            # 2D positional embed
            # TODO: this can be much simplified
            pos_embed = get_2d_sincos_pos_embed(
                self.transformer_dim, grid_size=(HH, WW)
            ).to(query_points.device) # 位置编码 维度=Transformer 维度 1 664 128 128
            sampled_pos_emb = sample_features4d(
                pos_embed.expand(B, -1, -1, -1), coords[:, 0]
            ) # 从位置编码中采样出与参考帧（coords[:, 0]）对应的编码信息。
            sampled_pos_emb = rearrange(
                sampled_pos_emb, "b n c -> (b n) c"
            ).unsqueeze(1) # (B*N, 1, transformer_dim)
            # # 在调用 get_1d_sincos_pos_embed_from_grid 之前，先把时间网格转到 CPU
            # time_grid = torch.linspace(0, S - 1, S, device='cpu').reshape(1, S, 1)
            # # 获取位置编码（在 CPU 上）
            # time_emb = get_1d_sincos_pos_embed_from_grid(self.transformer_dim, time_grid[0])
            # # 然后立即将结果转移到正确的设备上
            # time_emb = time_emb.to(query_points.device)

            # # 确保其他张量也在同一设备上
            # transformer_input = transformer_input.to(query_points.device)
            # sampled_pos_emb = sampled_pos_emb.to(query_points.device)

            # x = transformer_input + sampled_pos_emb + time_emb #比cotracker就少了一个time的编码
            x = transformer_input + sampled_pos_emb  # 比cotracker就少了一个time的编码

            # B, N, S, C 1 2048 8 664
            x = rearrange(x, "(b n) s d -> b n s d", b=B)

            # Compute the delta coordinates and delta track features
            delta = self.updateformer(x) #  1 2048 8 664-》 1 2048 8 130
            # BN, S, C
            delta = rearrange(delta, " b n s d -> (b n) s d", b=B)
            delta_coords_ = delta[:, :, :2] # 分离 Transformer 输出的更新信息
            delta_feats_ = delta[:, :, 2:]

            track_feats_ = track_feats_.reshape(B * N * S, self.latent_dim)
            delta_feats_ = delta_feats_.reshape(B * N * S, self.latent_dim)

            # Update the track features
            track_feats_ = (
                self.ffeat_updater(self.norm(delta_feats_)) + track_feats_
            )
            track_feats = track_feats_.reshape(
                B, N, S, self.latent_dim
            ).permute(
                0, 2, 1, 3
            )  # BxSxNxC 1 8 2048 128

            # B x S x N x 2    1 8 2048 2
            coords = coords + delta_coords_.reshape(B, N, S, 2).permute(
                0, 2, 1, 3
            )

            # Force coord0 as query
            # because we assume the query points should not be changed

            coords[:, 0] = coords_backup[:, 0] # 强制保持参考帧（通常为第一帧）的查询点坐标不变，因为这些坐标被视为固定的参考信息。
            # 你写在迭代里面可以保证。但是fine'tracker后面又保证了一次

            # The predicted tracks are in the original image scale

            if down_ratio > 1:
                coord_preds.append(coords * self.stride * down_ratio)
            else:
                coord_preds.append(coords * self.stride)

        # B, S, N
        if not self.fine: # 如果 self.fine 为 False，说明当前处于较粗的预测模式，此时需要计算可见性评分。
            vis_e = self.vis_predictor(
                track_feats.reshape(B * S * N, self.latent_dim)
            ).reshape(B, S, N) #  在M次迭代中不更新可见度，最后一次才更新可见度
            vis_e = torch.sigmoid(vis_e) # 使用 torch.sigmoid 将这些评分映射到 [0, 1] 之间，得到最终的可见性概率
        else:
            vis_e = None

        # 初始化 conf_e 为 None
        conf_e = None  # 在函数开始时初始化
        if self.cfg.track_conf: # 是否启用了 置信度预测 (track_conf 设置为 True)，如果启用，则会进行置信度的预测,
            # 这是一个简单的预测 后面对于精细会有更加精准的预测
            conf_e = self.conf_predictor(track_feats.reshape(B * S * N, self.latent_dim)).reshape(B, S, N)
            conf_e = torch.sigmoid(conf_e) # bsn

        if return_feat:
            return coord_preds, vis_e, track_feats, query_track_feat, conf_e # 因为只在这里面进行计算了
        # 粗略预测的所有轨迹点，预测的可见度， 所有轨迹的特征，查询帧的特征BSNC，可见的置信度（简单的线性层）
        else:
            return coord_preds, vis_e, conf_e
