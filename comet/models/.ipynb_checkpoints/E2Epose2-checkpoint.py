import json
import logging
import os
import sys
import traceback
from collections import defaultdict
from datetime import datetime

from pytorch3d.transforms import random_quaternions

from minipytorch3d.cameras import PerspectiveCameras
import torch.nn.functional as F

from accelerate.test_utils import training
from pytorch3d.renderer.cameras import CamerasBase
import torch.functional as F

from train_eval_func_new_cp5 import QuaternionCameras
from train_util import set_seed_and_print

from refine_track import refine_track  # 重新导入模块

from typing import Dict, List, Optional, Union

from accelerate import Accelerator
from omegaconf import OmegaConf, DictConfig

from losses import sequence_loss, balanced_ce_loss


logger = logging.getLogger(__name__)


import torch
import torch.nn as nn
from hydra.utils import instantiate
from typing import Dict


class TeacherForcingScheduler:
    def __init__(self, total_epochs, start_ratio=1.0, end_ratio=0.2, transition_epochs=300):
        self.total_epochs = total_epochs
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.transition_epochs = transition_epochs # 转变的地方

    def get_tf_ratio(self, epoch):
        """获取当前epoch的teacher forcing比例"""
        if epoch >= self.transition_epochs:
            return self.end_ratio
        return self.start_ratio - (self.start_ratio - self.end_ratio) * (epoch / self.transition_epochs)

    def should_use_teacher_forcing(self, epoch):
        """决定整个窗口是否使用teacher forcing"""
        ratio = self.get_tf_ratio(epoch)
        return torch.rand(1).item() < ratio


class COMET(nn.Module):
    def __init__(self, TRACK: Dict, CAMERA: Dict, cfg=None):
        super().__init__()

        # 检查 cfg 是否为 None
        if cfg is None:
            raise ValueError("cfg must be provided")

        self.cfg = cfg
        self.enable_track = cfg.enable_track  # 是否启用特征点追踪
        self.enable_pose = cfg.enable_pose    # 是否启用相机姿态估计
        self.window_len = cfg.window_len
        # self.stride = cfg.stride # 下采样的尺度

        if not (self.enable_track or self.enable_pose):
            raise ValueError("You have to enable at least tracking or pose estimation.")

        if self.enable_track:
            # 初始化 TrackerPredictor，包含 coarse 和 fine 追踪模块
            self.track_predictor = instantiate(TRACK, _recursive_=False, cfg=cfg)

            # self.coarse_fnet = self.track_predictor.coarse_fnet
            # self.coarse_predictor = self.track_predictor.coarse_predictor
            #
            # self.fine_fnet = self.track_predictor.fine_fnet
            # self.fine_predictor = self.track_predictor.fine_predictor

            # 根据配置冻结 track_predictor 中的参数
            self._freeze_tracker_params()

        if self.enable_pose:
            # 初始化相机预测器
            self.camera_predictor = instantiate(CAMERA, _recursive_=False, cfg=cfg)

    def _freeze_tracker_params(self):
        """
        根据 cfg 的设置冻结 track_predictor 的不同部分参数。
        freeze_ctrack: 冻结粗层模块（coarse_fnet 和 coarse_predictor），精细模块保持可训练。
        freeze_ftrack: 冻结精细模块（fine_fnet 和 fine_predictor），粗层模块保持可训练。
        freeze_track: 冻结整个 track_predictor 模块。
        """
        freeze_ctrack = getattr(self.cfg, "freeze_ctrack", False)
        freeze_ftrack = getattr(self.cfg, "freeze_ftrack", False)
        freeze_track  = getattr(self.cfg, "freeze_track", False)


        if freeze_track:
            for param in self.track_predictor.parameters():
                param.requires_grad = False

    def forward(
            self,
            image: torch.Tensor,  # 输入图像，形状为 (B, F, C, H, W)
            gt_cameras: Optional[CamerasBase] = None,  # 真实相机参数（如果有）
            training=True,  # 是否是训练模式
            tracks=None,  # 跟踪数据（如果有）groundtruth
            tracks_visibility=None,  # 特征点的可见性 一开始都是可见的 groundtruth # 1 8 2048
            crop_params=None,  # 裁剪参数（如果有）
            epoch=-1,# 训练的时候会用到
    ):
        if training:
           predictions = self.forward_all(
                                 image,
                                 gt_cameras=gt_cameras,
                                 training=training,
                                 tracks=tracks,
                                 tracks_visibility=tracks_visibility,
                                 )
        else:
            ##### %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #####
            #               EVALUATION
            ##### %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #####
            # 推理时建议确保不计算梯度：使用 torch.no_grad() 而非 no_grad
            batch_num = image.shape[0]  # 当前批次大小 bbb
            assert batch_num == 1, (
                f"评估模式下只能处理一个批次，"
                f"但当前 batch_num={batch_num}"
            )
            with torch.no_grad():
                predictions = self.forward_all(
                                 image,
                                gt_cameras=gt_cameras,
                                training=training,
                                tracks=tracks,
                                tracks_visibility=tracks_visibility,
                                 )

            # print("pred_pose_enc",predictions['pred_pose_enc'])
            # print("gt_pose_enc",predictions['gt_pose_enc'])
        return predictions


    def forward_all(self,
                  image: torch.Tensor,
                  preliminary_cameras: Optional[torch.Tensor] = None,
                  gt_cameras: Optional[CamerasBase] = None,
                  training: bool = True,
                  tracks: Optional[torch.Tensor] = None,
                  tracks_visibility: Optional[torch.Tensor] = None,
                 ) :

        B, T, C, H, W = image.shape
        if tracks is not None and tracks.numel() > 0:
            N = tracks.shape[2]  # 假设 tracks 形状为 [B, T, N, 2]
        predictions = {}

        ##### %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #####
        #               Training
        ##### %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #####
        loss = 0

        track_score = None
        fmaps_for_tracker = None

        pred_track = None
        track_confidence_input = None

        with torch.no_grad():
            if self.enable_track:  # 如果不冻结跟踪器
                ################先特征提取#######################
                # Prepare image feature maps for track_predictor 用的是cotracker 1 8 C=128 h=128 w=128
                fmaps_for_tracker = self.track_predictor.process_images_to_fmaps(image,training=training)
                ################先特征提取#######################

                if self.cfg.freeze_track:
                    torch.cuda.empty_cache()

                ##################粗跟踪#################
                # Coarse prediction
                coarse_pred_track_lists, vis_e, track_feats, query_track_feat, conf_e = self.track_predictor.coarse_predictor(
                    query_points=tracks[:, 0],  # B S N D-> B N D
                    fmaps=fmaps_for_tracker,  # [B, S, C , H, W]  image_size/8
                    iters=self.cfg.track_trainit,
                    down_ratio=self.track_predictor.coarse_down_ratio,
                    is_train=False,
                    return_feat=True,
                    TRACKorPOSE=False,
                )
                assert len(coarse_pred_track_lists) > 0, "Coarse predictor returned empty track list"
                coarse_pred_track = coarse_pred_track_lists[-1]  # 取列表中的迭代的最后一次作为最终的粗略预测轨迹 coarse_pred_track 1 8 2048 2
                ##################粗跟踪#################

                if self.cfg.freeze_track:
                    torch.cuda.empty_cache()

                ########细跟踪############
                if self.cfg.fine_tracker:  # 如果配置中启用了 fine tracker，则执行 refine_track 函数。
                    assert self.cfg.train.fix_first_cor
                    refined_tracks, track_score = refine_track(
                        image,  # 1 8 3 1024 1024
                        self.track_predictor.fine_fnet,
                        self.track_predictor.fine_predictor,
                        coarse_pred_track,  # 传入最后一次迭代的结果
                        compute_score=True,
                    )

                    if self.cfg.freeze_track:
                        torch.cuda.empty_cache()
                else:
                    refined_tracks = coarse_pred_track  # 若不进行细跟踪
                    track_score = torch.ones_like(vis_e) # 得分为1
                ########细跟踪############


                if self.cfg.softmax_refine or self.cfg.fine_tracker:
                    # 正确的方式
                    pred_tracks = coarse_pred_track_lists + [refined_tracks]
                    # 将优化之后的轨迹加入到预测的轨迹中去：几次粗跟踪的结果的后面加上细跟粽的结果
                else:
                    pred_tracks = coarse_pred_track_lists  #

                pred_track = pred_tracks[-1]  # 最终的轨迹

                if self.cfg.fine_tracker and track_score is not None:
                    eps = 1e-6
                    inverted_score = 1.0 / (track_score + eps)  # [B, S, N]
                    # 归一化处理
                    inverted_score = inverted_score / inverted_score.max(dim=1, keepdim=True)[0]
                    predictions["coarse_pred_track"] = coarse_pred_track
                    predictions["refine_pred_track"] = pred_track
                    predictions["pred_score"] = inverted_score # 归一化之后的得分，越高越好
            #################################统计轨迹 损失###########################


        if self.enable_pose:
            # 确定使用哪个置信度值
            if self.cfg.fine_tracker and track_score is not None:
                track_confidence_input = inverted_score  # 使用转换后的细跟踪score

            pose_predictions = self.camera_predictor(
                image.reshape(-1,C,H,W),
                preliminary_cameras=None,
                batch_size=B,
                gt_cameras=gt_cameras,
                iters=self.cfg.camera_iter,
                # fmaps=fmaps_for_tracker,
                pred_trajectories=pred_track,
                track_confidence=track_confidence_input,  # 使用选择的置信度
            )

        # 在这里添加GPU内存清理
        if not training:
            torch.cuda.empty_cache()

        if self.enable_track:
            pose_predictions["pred_tracks"] = predictions["refine_pred_track"]

        return pose_predictions


    def forward_window(self,
                  image: torch.Tensor,
                  preliminary_cameras: Optional[torch.Tensor] = None,
                  batch_size: int = 1,
                  gt_cameras: Optional[CamerasBase] = None,
                  iters: int = 4,
                  training: bool = True,
                  tracks: Optional[torch.Tensor] = None,
                  tracks_visibility: Optional[torch.Tensor] = None,
                  crop_params: Optional[torch.Tensor] = None,
                  epoch=-1,
                 ) :
        """
        滑动窗口方式的前向传播
        Args:
            image (torch.Tensor): 输入图像, shape [B, T, C, H, W]
            preliminary_cameras: 初始相机参数
            batch_size: 批次大小
            gt_cameras: Ground truth 相机参数
            iters: 迭代次数
            training: 是否为训练模式
            tracks: 跟踪ground truth  这个必须得有
            tracks_visibility: 特征点可见性ground truth
            crop_params: 裁剪参数
        """
        loss = 0
        B, T, C, H, W = image.shape
        N = tracks.shape[2]  # 假设 tracks 形状为 [B, T, N, 2]
        predictions = {}

        S = self.window_len
        assert S > 1, "Window length must be greater than 1"
        step = max(1, S // 2)  # 确保步长至少为1

        pad = (S - T % S) % S
        length = S-pad
        if pad > 0:
            if pad <= S // 2:
                image_pad = image[:, -pad:].flip(1)
            else:
                # 对于较大的填充，结合镜像和循环填充
                pad1 = S // 2
                pad2 = pad - pad1
                image_pad = torch.cat([
                    image[:, -pad1:].flip(1),
                    image[:, :pad2]
                ], dim=1)
            image = torch.cat([image, image_pad], dim=1)
        T_padded = T + pad
        num_windows = (T_padded - S) // step + 1
        indices = list(range(0, num_windows * step, step))
        fmaps_for_tracker = None

        # 4. 初始化结果张量
        device = image.device
        all_pred_track = torch.zeros((B, T_padded, N, 2), device=device)
        all_track_score = torch.zeros((B, T_padded, N), device=device)
        all_track_vis = torch.zeros((B, T_padded, N), device=device)
        all_track_con = torch.zeros((B, T_padded, N), device=device)

        # 在循环外初始化所有累计损失值
        total_seq_loss = torch.tensor(0., device=image.device)
        total_vis_loss = torch.tensor(0., device=image.device)
        total_conf_loss = torch.tensor(0., device=image.device)
        total_refine_loss = torch.tensor(0., device=image.device)


        if self.enable_track:  # 如果不冻结跟踪器

            tf_scheduler = TeacherForcingScheduler(
                total_epochs=self.cfg.train.epochs,
                start_ratio=self.cfg.train.teacher_forcing.start_ratio,
                end_ratio=self.cfg.train.teacher_forcing.end_ratio,
                transition_epochs=self.cfg.train.teacher_forcing.transition_epochs
            ) #todo:新加入的

            ################先特征提取#######################
            # Prepare image feature maps for track_predictor 用的是cotracker 1 8 C=128 h=128 w=128
            fmaps_for_tracker = self.track_predictor.process_images_to_fmaps(image)
            ################先特征提取#######################

            for ind in indices:
                # 当前窗口内的图像及特征： [B, S, ...]
                end_ind_length = 0
                if ind+S>T:
                    end_ind_length = length
                window_images = image[:, ind: ind + S]
                window_fmaps = fmaps_for_tracker[:, ind: ind + S]

                if ind == 0:
                    # 第一个窗口始终使用第一帧的 真实轨迹初始化
                    track_init = tracks[:, 0].reshape(B, 1, N, 2).repeat(1, S, 1, 1)
                else:
                    # 对于后续窗口，整个窗口统一使用真实值或预测值 todo:新增的
                    use_teacher_forcing = tf_scheduler.should_use_teacher_forcing(epoch) if training else False
                    # tf_ratio = tf_scheduler.get_tf_ratio(epoch)

                    if use_teacher_forcing and tracks is not None:
                        # 整个窗口使用真实轨迹
                        overlap_tracks = tracks[:, ind:ind + step].clone()
                    else:
                        # 整个窗口使用预测轨迹
                        overlap_tracks = all_pred_track[:, ind - step:ind].clone()
                    last_track = overlap_tracks[:, -1:].clone()
                    num_to_fill = S - step
                    fill_tracks = last_track.expand(-1, num_to_fill, -1, -1)
                    track_init = torch.cat([overlap_tracks, fill_tracks], dim=1)

                # 提取当前窗口对应的 Ground Truth 和可见性信息（若提供）
                gt_track_window = tracks[:, ind: ind + S] if tracks is not None else None
                gtvis_track_window = tracks_visibility[:, ind: ind + S] if tracks_visibility is not None else None

                ################ 粗跟踪预测 ################
                coarse_pred_track_lists, vis_e, track_feats, query_track_feat, conf_e = self.coarse_predictor(
                    query_points=track_init,  # 初始查询点为上一步指定的track_init
                    fmaps=window_fmaps,  # 当前窗口内的特征图 [B, S, C, H, W]
                    iters=self.cfg.track_trainit,
                    down_ratio=self.track_predictor.coarse_down_ratio,
                    is_train=training,
                    return_feat=True,
                    ind=ind,
                )
                # coarse_pred_track_lists 是每次迭代的结果，取最后一次迭代作为粗跟踪结果
                coarse_pred_track = coarse_pred_track_lists[-1]  # shape: (B, S, N, 2)

                ################ 细跟踪（若启用 fine track_predictor） ################
                refine_loss = None
                if self.cfg.fine_tracker:
                    # 传入当前窗口图像、粗跟踪结果以及对应 GT 和可见性信息进行细化
                    full_refined_tracks, refine_loss, refined_tracks, track_score = refine_track(
                        window_images,  # (B, S, 3, H, W)
                        self.fine_fnet,
                        self.fine_predictor,
                        coarse_pred_track,  # 粗跟踪结果传入细化模块
                        gt_track_window,
                        gtvis_track_window,
                        coarse_fmaps=window_fmaps,
                        is_train=training,
                        compute_score=True,
                        cfg=self.cfg,
                        conf=conf_e,
                        ind=ind,
                    )
                else:
                    full_refined_tracks = coarse_pred_track  # 若不进行细跟踪
                    track_score = torch.zeros((B, S, N), device=image.device)

                # 将当前窗口内的预测结果存入 pred_track（后续可能作为下个窗口的初始值使用）
                all_pred_track[:, ind: ind + S] = full_refined_tracks
                all_track_score[:, ind: ind + S]  = track_score
                all_track_vis[:, ind: ind + S]  = vis_e
                if self.cfg.track_conf:
                    all_track_con[:, ind: ind + S] = conf_e

                ################ 损失计算 ################
                if gt_track_window is not None:
                    # 构建有效性掩膜：默认所有轨迹均有效，但可过滤掉第一帧不合理的轨迹
                    valids = torch.ones(gtvis_track_window.shape, device=gtvis_track_window.device)
                    mask = gtvis_track_window[:, 0, :]  # 以窗口内第一帧为基准过滤
                    valids = valids * mask.unsqueeze(1)
                    ignore_first = True if ind==0 else False
                    # 轨迹序列损失，衡量预测轨迹与 GT 的误差
                    seq_loss = sequence_loss(
                        coarse_pred_track_lists,
                        gt_track_window,
                        gtvis_track_window,
                        valids, 0.8,
                        vis_aware=self.cfg.train.vis_aware,
                        vis_aware_w=self.cfg.train.vis_aware_w,
                        huber=self.cfg.train.huber,
                        max_thres=self.cfg.clip_trackL,
                        ignore_first=ignore_first,
                    )
                    if not torch.isfinite(seq_loss).all():
                        raise RuntimeError("seq_loss contains NaN or Inf values")  # 改为抛出异常而不是打印警告
                    # 修改后的代码
                    # 在计算 vis_loss 之前添加形状检查
                    print("\n=== Visibility Loss Input Shapes ===")
                    print(f"vis_e shape: {vis_e.shape}")
                    print(f"tracks_visibility slice shape: {tracks_visibility[:, ind: ind + S].shape}")
                    print(f"valids shape: {valids.shape}")

                    # 确保形状匹配
                    assert vis_e.shape == tracks_visibility[:, ind: ind + S].shape == valids.shape, (
                        f"Shape mismatch in visibility loss calculation:\n"
                        f"vis_e: {vis_e.shape}\n"
                        f"tracks_visibility: {tracks_visibility[:, ind: ind + S].shape}\n"
                        f"valids: {valids.shape}"
                    )

                    # 可见性损失，通过交叉熵衡量预测 vis_e 与 GT 的差异
                    vis_loss, _ = balanced_ce_loss(vis_e, tracks_visibility[:, ind: ind + S], valids)
                    if not torch.isfinite(vis_loss).all():
                        print("Warning: vis_loss contains NaN or Inf.")
                    # 跟踪置信度损失
                    if self.cfg.track_conf:
                        final_dis = torch.sqrt(torch.sum((coarse_pred_track - gt_track_window) ** 2, dim=-1))
                        conf_loss, _ = balanced_ce_loss(conf_e, final_dis < 1, valids)
                    else:
                        conf_loss = vis_loss * 0

                    # 防止 refine_loss 未定义
                    if refine_loss is None:
                        refine_loss = seq_loss * 0

                    total_seq_loss = total_seq_loss + seq_loss
                    total_vis_loss = total_vis_loss + vis_loss
                    if self.cfg.track_conf:
                        total_conf_loss = total_conf_loss + conf_loss
                    total_refine_loss = total_refine_loss + refine_loss


                ################ 汇总跟踪分支损失 ################
            loss_for_tracking = (total_seq_loss + total_vis_loss * 10 + total_conf_loss * 10 + total_refine_loss)
            loss_for_tracking = loss_for_tracking * self.cfg.train.track_weight
            loss = loss + loss_for_tracking  # 将跟踪损失累加到总损失中


            true_pred_track = all_pred_track[:, :T]
            true_pred_vis = all_track_vis[:, :T]
            true_pred_score = all_track_score[:, :T]

            if true_pred_track[:,0]==tracks[:,0]:
                print("这下对了###################（（（（（（（（（（（（（（（（%%%%%%%%%%%%%")
            else:
                print("true_pred_track[:,0]true_pred_track[:,0]true_pred_track[:,0]")

            ################################################################################
            # # # # #
            # TODO: 检查这里的逻辑，强制将 padding 区域的预测设为非可见
            if crop_params is not None:  # 检查是否提供了crop_params，如果提供了，则进入条件语句。
                boundaries = crop_params[:, :, -4:-2].abs()  # 从crop_params中提取倒数第4列和倒数第3列的数据，并取其绝对值，得到boundaries
                boundaries = torch.cat([boundaries, image.shape[-1] - boundaries], dim=-1)
                # 将原始边界和剩余距离在最后一个维度上拼接起来，得到一个包含四个数值的张量，这四个数值通常就代表了裁剪区域的完整边界：例如左、上、右、下的坐标。
                final_pred = true_pred_track  # 获取pred_tracks中的最后一次迭代的预测结果，赋值给final_pred。
                hvis = torch.logical_and(
                    final_pred[..., 1] >= boundaries[:, :, 1:2], final_pred[..., 1] <= boundaries[:, :, 3:4]
                )  # 计算final_pred在第1维度y 上的值是否在boundaries的第1列和第3列之间，返回布尔张量hvis
                wvis = torch.logical_and(
                    final_pred[..., 0] >= boundaries[:, :, 0:1], final_pred[..., 0] <= boundaries[:, :, 2:3]
                )  # 计算final_pred在第0维度上的值是否在boundaries的第0列和第2列之间，返回布尔张量wvis。
                force_vis = torch.logical_and(hvis, wvis)
                true_pred_vis = true_pred_vis * force_vis.float()

            ################ 存储预测结果 ################
            predictions["loss_track"] = total_seq_loss
            # predictions["loss_track"] *= (1 + (1 - tf_ratio))  # todo:随着真实值依赖减少，增加轨迹损失权重
            predictions["loss_vis"] = total_vis_loss * 10
            predictions["loss_tconf"] = total_conf_loss * 10
            predictions["loss_re"] = total_refine_loss
            predictions["pred_tracks"] = true_pred_track  # 去除填充部分,直接就是最后一个迭代的轨迹
            predictions["pred_vis"] = true_pred_vis   # 记录最后一个窗口的可见性预测
            if self.cfg.fine_tracker and track_score is not None:
                predictions["pred_score"] = true_pred_score
        if self.enable_pose:  # 如果不冻结跟踪器
            pose_predicted = torch.zeros((B, T_padded, 7), device=image.device)  # 存储姿态预测结果
            preliminary_cameras = None

            for ind in indices:
                # 当前窗口图像
                window_images = image[:, ind: ind + S].reshape(-1,C,H,W)

                if ind == 0:
                    # 第一个窗口使用零初始化（或其他初始化方式）
                    pose_init = torch.zeros(B, S, self.target_dim, device=image.device)
                    quat_init = torch.tensor([0, 0, 0, 1], device=image.device)
                    quat_init = F.normalize(quat_init, dim=0)  # 确保单位四元数
                    pose_init[:, :, 3:] = quat_init
                else:
                    use_teacher_forcing = tf_scheduler.should_use_teacher_forcing(epoch) if training else False
                    # 在损失计算部分添加基于teacher forcing比例的权重调整
                    tf_ratio = tf_scheduler.get_tf_ratio(epoch)

                    if use_teacher_forcing and gt_cameras is not None:
                        # 使用真实值
                        overlap_poses = gt_cameras[:, ind - step:ind].clone()
                    else:
                        # 对于后续窗口，先取出前一窗口最后 step 帧作为重叠部分
                        overlap_poses = pose_predicted[:, ind - step:ind].clone()  # (B, step, 7)
                    # 取重叠部分最后一帧，并复制 (S - step) 次
                    last_pose = overlap_poses[:, -1:].clone()  # (B, 1, 7)
                    num_to_fill = S - step
                    fill_poses = last_pose.expand(-1, num_to_fill, -1)  # (B, S - step, 7)
                    # 拼接，构成 (B, S, 7) 的初始化
                    pose_init = torch.cat([overlap_poses, fill_poses], dim=1)

                # 处理 Ground Truth（仅在提供 GT 时）
                if gt_cameras is not None:
                    gt_window = gt_cameras[:, ind: ind + S]
                else:
                    gt_window = None

                # 预测相机姿态
                if self.cfg.fine_tracker:  # 如果启用了细跟踪
                    # 使用细跟踪得到的score作为置信度
                    track_confidence_input = all_track_score[:, ind: ind + S]
                elif self.cfg.track_conf:
                    # 如果没有启用细跟踪，使用粗跟踪的置信度
                    track_confidence_input = all_track_con[:, ind: ind + S]
                else:
                    # 如果既没有细跟踪也没有置信度预测，抛出错误
                    raise ValueError(
                        "\n错误: 无法获取跟踪置信度！"
                        "\n请确保满足以下条件之一:"
                        "\n1. 启用细跟踪 (fine_tracker=True)"
                        "\n2. 启用跟踪置信度预测 (track_conf=True)"
                        "\n当前配置:"
                        f"\n- fine_tracker: {self.cfg.fine_tracker}"
                        f"\n- track_conf: {self.cfg.track_conf}"
                        "\n\n建议:"
                        "\n- 如需使用细跟踪，请在配置文件中设置 fine_tracker=True"
                        "\n- 如需使用粗跟踪的置信度，请在配置文件中设置 track_conf=True"
                    )


                # 预测相机姿态
                pose_predictions = self.camera_predictor(
                    window_images,
                    preliminary_cameras=pose_init,
                    batch_size=B,
                    gt_cameras=gt_window,
                    iters=iters,
                    fmaps=fmaps_for_tracker,
                    pred_trajectories=all_pred_track[:, ind: ind + S],
                    track_confidence=track_confidence_input ,
                )

                pose_predicted[:, ind: ind + S] = pose_predictions

            predictions["preliminary_cameras"] = preliminary_cameras

            predictions["pred_cameras"] = pose_predictions["pred_cameras"] # 这是一个类（通常用于处理相机模型）
            predictions["loss_pose"] = pose_predictions["loss_pose"]
            # predictions["loss_pose"] *= (1 + (1 - tf_ratio)) # todo:新增的 你看看 不要这里了
            loss_for_pose = predictions["loss_pose"]
            loss = loss + loss_for_pose

        predictions["loss"] = loss

        # 在这里添加GPU内存清理
        if not training:
            torch.cuda.empty_cache()

        return predictions

def build_comet_model(self):
    comet = instantiate(self.cfg.MODEL, _recursive_=False, cfg=self.cfg)
    
    if self.cfg.auto_download_ckpt:
        comet.from_pretrained(self.cfg.model_name)
    else:
        checkpoint = torch.load(self.cfg.resume_ckpt) 
        comet.load_state_dict(checkpoint, strict=True)
        
    self.comet_model = comet.to(self.device).eval()
    print("COMET built successfully")



import torch

def simulate_tracks(B, S, N, H, W, step_sigma=5.0, device="cuda"):
    """
    生成形状为 (B, S, N, 2) 的轨迹坐标张量。
    - B: batch size
    - S: 时间步数（帧数）
    - N: 每帧的轨迹点数量
    - H, W: 图像高和宽
    - step_sigma: 每帧坐标移动的高斯标准差（像素）
    """
    # 初始化：在图像内均匀随机采样 N 个起始点
    # 坐标顺序为 (x, y)，范围分别为 [0, W-1], [0, H-1]
    init_x = torch.rand(B, N, device=device) * (W - 1)
    init_y = torch.rand(B, N, device=device) * (H - 1)
    # shape -> (B, 1, N, 2)
    tracks = torch.stack([init_x, init_y], dim=-1).unsqueeze(1).repeat(1, S, 1, 1)

    # 对后续帧执行随机漫步
    # 每一步在 x/y 方向上分别加入零均值、std=step_sigma 的高斯噪声
    for t in range(1, S):
        # 采样移动量
        delta = torch.randn(B, N, 2, device=device) * step_sigma
        # 叠加到上一帧
        tracks[:, t] = tracks[:, t - 1] + delta
        # Clamp 保证落在图像范围内
        tracks[:, t, :, 0].clamp_(0, W - 1)
        tracks[:, t, :, 1].clamp_(0, H - 1)

    return tracks


def main(cfg: DictConfig):
    #######################1. 配置初始化#############################
    OmegaConf.set_struct(cfg, False)

    # 设置详细的日志记录
    logging.basicConfig(
        level=logging.INFO if not cfg.debug else logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'model_debug_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    logger = logging.getLogger(__name__)

    accelerator = Accelerator(even_batches=False, device_placement=False, mixed_precision=cfg.mixed_precision)

    logger.info("=" * 50)
    logger.info("环境配置信息：")
    logger.info(f"PyTorch 版本: {torch.__version__}")
    logger.info(f"CUDA 是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA 版本: {torch.version.cuda}")
        logger.info(f"当前GPU设备: {torch.cuda.get_device_name()}")
        logger.info(f"GPU数量: {torch.cuda.device_count()}")
    logger.info(f"混合精度模式: {cfg.mixed_precision}")
    logger.info("=" * 50)

    accelerator.print("Model Config:")
    accelerator.print(OmegaConf.to_yaml(cfg))
    accelerator.print(accelerator.state)

    ##########################设备与调试模式########################
    if cfg.debug:
        logger.info("********DEBUG MODE********")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = cfg.train.cudnnbenchmark

    ##############################3. 设定随机种子 保证可复现性###############
    set_seed_and_print(cfg.seed)

    ################ 随机生成输入数据######################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 相机参数设置 [SKIP] weight
    B, S, C, H, W = 1, 24, 3, 256, 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 随机图像
    image = torch.rand(B, S, C, H, W, device=device)

    # 1. 随机四元数旋转 R ∈ [B, S, 4]
    quat = random_quaternions(B * S, device=device).view(B, S, 4)

    # 2. 平移向量 T ∈ [B, S, 3]
    T = (torch.rand(B, S, 3, device=device) * 2.0 - 1.0)  # [-1, 1] 区间

    # 3. 焦距 focal length ∈ [B, S, 2]
    focal_length = torch.full((B, S, 2), 1000.0, device=device)  # fx=fy=1000

    # 4. 主点 principal point ∈ [B, S, 2]
    principal_point = torch.full((B, S, 2), 0.0, device=device)  # 默认 (0, 0)，可设为图像中心

    # 5. 构造四元数相机
    gt_cameras = QuaternionCameras(
        focal_length=focal_length.reshape(B * S, 2),
        principal_point=principal_point.reshape(B * S, 2),
        R=quat.reshape(B * S, 4),
        T=T.reshape(B * S, 3),
        device=device
    )
    # 其他输入数据
    tracks = simulate_tracks(B=B, S=S, N=10, H=H, W=W, step_sigma=5.0, device=device)
    tracks_visibility = torch.randint(0, 2, (B, S, 10), device=device).bool()

    # 实例化模型并移至GPU
    logger.info("\n开始实例化模型...")
    model = instantiate(cfg.MODEL, _recursive_=False, cfg=cfg)

    # 1) 加载检查点
    checkpoint = torch.load(cfg.train.resume_ckpt)  # 从本地磁盘加载 PyTorch 检查点文件

    # 2) 获取当前模型的 state_dict
    model_dict = model.state_dict()

    # 3) 筛选出 checkpoint 中 track_predictor 部分的权重
    pretrained_track = {
        k: v
        for k, v in checkpoint.items()
        if k.startswith("track_predictor.")  # 仅筛选以 "track_predictor." 开头的键
    }

    # 4) 更新到当前模型的 state_dict
    # 截取的 track_predictor 参数会覆盖当前模型中已有的参数
    model_dict.update(pretrained_track)

    # 5) 按非严格模式加载（只加载匹配的权重，忽略不匹配的参数）
    model.load_state_dict(model_dict, strict=True)

    # 将模型移动到设备（如 GPU 或 CPU）
    model = model.to(device)

    pose_params = list(model.camera_predictor.parameters())

    # 构建 AdamW 优化器，仅更新姿态估计的权重
    optimizer = torch.optim.AdamW(
        params=pose_params,
        lr=cfg.train.lr
    ) 
    logger.info("\n开始训练模式前向传播测试...")
    model.train()
    try:
        with torch.cuda.amp.autocast(enabled=cfg.mixed_precision != "no"):  # 启用混合精度训练
            outputs = model(image, gt_cameras=gt_cameras, training=True,
                                  tracks=tracks, tracks_visibility=tracks_visibility)

        loss = outputs["loss"].mean()
        # —— 反向 & 优化 ——
        optimizer.zero_grad()


        # 标准 FP32 反向
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1
        )
        optimizer.step()

        logger.info("训练模式输出:")
        # for key, value in outputs_train.items():
        #     if isinstance(value, torch.Tensor):
        #         logger.info(f"{key}:")
        #         logger.info(f"    形状: {value.shape}")
        #         logger.info(f"    数据类型: {value.dtype}")
        #         logger.info(f"    设备: {value.device}")
        #         logger.info(f"    数值范围: [{value.min().item():.3f}, {value.max().item():.3f}]")
        #         if torch.isnan(value).any():
        #             logger.warning(f"    警告: {key} 包含 NaN 值!")
        #         if torch.isinf(value).any():
        #             logger.warning(f"    警告: {key} 包含 Inf 值!")
        #     elif isinstance(value, PerspectiveCameras):  # 添加对相机对象的特殊处理
        #         logger.info(f"{key}: PerspectiveCameras object")
        #         # logger.info(f"    focal_length: {value.focal_length}")
        #         # logger.info(f"    R: {value.R}")
        #         # logger.info(f"    T: {value.T}")
        #     else:
        #         logger.info(f"{key}: {type(value)}")
    except Exception as e:
        logger.error("前向传播出错:")
        logger.error(traceback.format_exc())

    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("=" * 50)
    logger.info("模型分析完成")
    logger.info("=" * 50)


if __name__ == '__main__':
    cfg = OmegaConf.load('train.yaml')
    main(cfg)

