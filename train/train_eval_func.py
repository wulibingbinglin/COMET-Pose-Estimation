import argparse
import cProfile
import datetime
import glob
import io
import json
import os
import pickle
import pstats
import re
import time
from collections import OrderedDict
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Union
from visdom import Visdom

# Related third-party imports
from accelerate import Accelerator, DistributedDataParallelKwargs, GradScalerKwargs
from hydra.utils import instantiate, get_original_cwd
from torch.cuda.amp import GradScaler, autocast
from train_util import process_co3d_data

import cv2
import hydra
# import models
import numpy as np
import psutil
import torch
import tqdm
import visdom
from omegaconf import OmegaConf, DictConfig
from pytorch3d.implicitron.tools import model_io, vis_utils
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import corresponding_cameras_alignment
from pytorch3d.ops.points_alignment import iterative_closest_point, _apply_similarity_transform
from pytorch3d.renderer.cameras import PerspectiveCameras, FoVPerspectiveCameras
from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import plot_scene

# from griddle.utils import is_aws_cluster
# from test_category import test_co3d, test_imc
# from util.load_img_folder import load_and_preprocess_images
# from util.metric import camera_to_rel_deg, calculate_auc, calculate_auc_np, camera_to_rel_deg_pair
# from util.triangulation import intersect_skew_line_groups
# from train_util import *
# from inference import run_inference


def train_or_eval_fn(
    model, dataloader, cfg, optimizer, stats, accelerator, lr_scheduler, training=True, viz=None, epoch=-1
):
    if training: # 根据 training 参数，判断是否进入训练模式。
        model.train()
    else:
        model.eval() #

    time_start = time.time()
    max_it = len(dataloader) # 每个 epoch 中的总步数（batch 数量）

    if cfg.track_by_spsg: # 这个条件判断是否使用 SuperPoint 或其他特征提取器来处理关键点。cfg.track_by_spsg 是一个配置项，决定是否启用这些特征提取器。
        # 这些行导入了一些特征提取器模块
        from extractors.superpoint_open import SuperPoint #
        from extractors.sift import SIFT

        # 初始化 SuperPoint 特征提取器，cuda() 将其移到 GPU，eval() 设置为评估模式。相同的操作也适用于 DISK, SIFT
        sp = SuperPoint({"nms_radius": 4, "force_num_keypoints": True}).cuda().eval()
        sift = SIFT({}).cuda().eval()

    AUC_scene_dict = {} # 初始化一个空字典，通常用于记录与 AUC 相关的场景数据。
    
    for step, batch in enumerate(dataloader): # 进入数据加载循环，逐步处理 dataloader 中的每个批次（batch）。
        # print(batch["seq_name"])
        if step == 100: #X 当迭代到第 100 个批次时，记录并打印当前的 CPU 内存使用情况。这通常用于调试。
            record_and_print_cpu_memory_and_usage()

        images_hd = None

        gt_cameras = None # 初始化 gt_cameras 为 None，用于存储地面真实值的相机信息。
        points_rgb = None

        #############先进行数据处理########
        (
            images,
            crop_params,
            translation,
            rotation,
            fl,
            pp,
            points,
            points_rgb,
            tracks,
            tracks_visibility,
            tracks_ndc,
        ) = process_co3d_data(batch, accelerator.device, cfg)
        # 调用 process_co3d_data 函数来处理当前批次的数据。这个函数返回一组图像和其他相关数据，包括图像、裁剪参数、平移、旋转、焦距、相机内参、点云、点云 RGB 信息、轨迹

        ######### 特征提取器的初始化、关键点的选择和筛选： 通过合并不同的特征提取器的输出并筛选出数量最合适的关键点，用于后续的轨迹跟踪。轨迹的创建：##########
        if cfg.track_by_spsg and (not cfg.inference): # 判断是否启用了 track_by_spsg（即启用了特征提取器），并且当前不是推理阶段（cfg.inference 为 False）。
            # use keypoints as the starting points of tracks
            images_for_kp = images # 将图像数据赋值给 images_for_kp，这是用于提取关键点的数据
            bbb, nnn, ppp, _ = tracks.shape # bbb 是批次大小，nnn 是轨迹数量，ppp 是每个轨迹的维度。

            pred0_sp = sp({"image": images_for_kp[:, 0]}) # 使用 SuperPoint 提取器提取 第一帧 图像的关键点。
            kp0_sp = pred0_sp["keypoints"] # 从 SuperPoint 的输出中提取关键点。
            pred0_sift = sift({"image": images_for_kp[:, 0]}) # 使用 SIFT 特征提取器提取第一帧图像的关键点。
            kp0_sift = pred0_sift["keypoints"] # 从 Sift 的输出中提取关键点。
            kp0 = torch.cat([kp0_sp, kp0_sift], dim=1) #从将 SuperPoint 和 SIFT 提取的关键点合并（按维度 1 拼接）。

            new_track_num = kp0.shape[-2] #从获取合并后的关键点数量。
            if new_track_num > cfg.train.track_num: # 如果关键点数量超过了预设的最大数量（cfg.train.track_num），则随机选择一部分关键点。
                indices = torch.randperm(new_track_num)[: cfg.train.track_num] # 随机选择前 cfg.train.track_num 个关键点。
                kp0 = kp0[:, indices, :] # 将 kp0 限制为选择的关键点。

            # print(kp0)
            kp0 = kp0[None].repeat(1, nnn, 1, 1) # 扩展 kp0 的维度，使其适应批处理中的每个批次的每帧的每个轨迹。
            tracks = kp0.clone()
            tracks_visibility = torch.ones(bbb, nnn, kp0.shape[-2]).bool().cuda().clone()
            #从初始化轨迹可见性（tracks_visibility）为 True，表示所有的轨迹都是可见的。

        frame_size = images.shape[1] #从获取图像的帧大小，帧数

        
        if rotation is not None: # 相机和数据处理：
            # 如果 rotation（旋转矩阵）不是 None，即模型的旋转矩阵存在，那么我们将根据提供的旋转矩阵、平移向量以及焦距、主点等信息来创建相机
            gt_cameras = PerspectiveCameras(
                focal_length=fl.reshape(-1, 2), #焦距
                principal_point=pp.reshape(-1, 2),# 主点
                R=rotation.reshape(-1, 3, 3),
                T=translation.reshape(-1, 3),
                device=accelerator.device,
            )
            batch_size = len(images) #从获取当前批次（images）的大小，用于后续的训练或评估。


            if training and cfg.train.batch_repeat > 0:
                # 如果是训练模式，并且配置中设置了批次重复（cfg.train.batch_repeat > 0），则将当前批次的样本重复多次，以加速训练。
                # repeat samples by several times
                # to accelerate training
                br = cfg.train.batch_repeat # 获取批次重复的次数。
                gt_cameras = PerspectiveCameras(
                    focal_length=fl.reshape(-1, 2).repeat(br, 1),
                    R=rotation.reshape(-1, 3, 3).repeat(br, 1, 1),
                    T=translation.reshape(-1, 3).repeat(br, 1),
                    device=accelerator.device,
                ) #v将相机参数（焦距、旋转矩阵、平移向量）按批次重复多次，以增加训练数据的大小。
                batch_size = len(images) * br # 更新批次大小为原始批次的大小乘以重复次数。



        if training: # 进入训练模式。如果是训练，我们使用模型进行前向传播。
            predictions = model(
                images,
                gt_cameras=gt_cameras,
                training=True,
                # batch_repeat=cfg.train.batch_repeat,
                points=points,
                points_rgb=points_rgb,
                tracks=tracks,
                tracks_visibility=tracks_visibility,
                tracks_ndc=tracks_ndc,
                epoch=epoch,
                crop_params=crop_params,
            )

            predictions["loss"] = predictions["loss"].mean()
            loss = predictions["loss"]
        else:
            # for _ in range(10):
            #     print("GOOOOOOOOOOOOO")
            imgpaths = None
            if cfg.inference: # 如果配置中设置了 inference=True，即是推理模式
                predictions = run_inference(
                    model,
                    images,
                    gt_cameras=gt_cameras,
                    training=False,
                    points=points,
                    points_rgb=points_rgb,
                    tracks=tracks,
                    tracks_visibility=tracks_visibility,
                    tracks_ndc=tracks_ndc,
                    epoch=epoch,
                    imgpaths=imgpaths,
                    crop_params=crop_params,
                    images_hd=images_hd,
                    batch=batch,
                    cfg=cfg,
                )
            else:
                with torch.no_grad():
                    # 如果不是推理模式（即评估模式），则调用模型进行前向传播。使用 torch.no_grad() 来禁止梯度计算，因为在评估模式下我们不需要反向传播。
                    predictions = model(
                        images,
                        gt_cameras=gt_cameras,
                        training=False,
                        points=points,
                        points_rgb=points_rgb,
                        tracks=tracks,
                        tracks_visibility=tracks_visibility,
                        tracks_ndc=tracks_ndc,
                        epoch=epoch,
                        imgpaths=imgpaths,
                        crop_params=crop_params,
                        images_hd=images_hd,
                        batch=batch,
                    )

        # Computing Metrics 用于 计算评估指标 评估预测的相机姿态 (pred_cameras) 与真实相机姿态 (gt_cameras) 之间的误差。
        with torch.no_grad(): # 使用 torch.no_grad() 来减少内存占用并加速计算。
            if "pred_cameras" in predictions: # 检查 predictions 字典中是否包含 pred_cameras（即模型是否预测了相机参数）
                with autocast(dtype=torch.double):
                # 使用 autocast(dtype=torch.double) 以 double (64-bit) 精度进行计算，提高计算精度，特别是在涉及旋转矩阵和角度计算时
                    pred_cameras = predictions["pred_cameras"]

                    rel_rangle_deg, rel_tangle_deg = camera_to_rel_deg(
                        pred_cameras, gt_cameras, accelerator.device, batch_size
                    ) # 计算相对旋转角误差 (rel_rangle_deg) 和相对平移角误差 将旋转矩阵转换为角度误差，以及 计算平移向量的角度误差

                    # metrics to report
                    thresholds = [5, 15, 30]
                    # 计算准确率
                    for threshold in thresholds:
                        predictions[f"Racc_{threshold}"] = (rel_rangle_deg < threshold).float().mean() # 计算 Racc_5, Racc_15, Racc_30（旋转准确率）
                        predictions[f"Tacc_{threshold}"] = (rel_tangle_deg < threshold).float().mean() # # 结果: (1+0+1+0+0)/5 = 0.4

                    # 计算 AUC（累积误差分布）  累计误差曲线 (AUC, Area Under Curve)，用来衡量模型的整体表现。
                    Auc_30, normalized_histogram = calculate_auc(
                        rel_rangle_deg, rel_tangle_deg, max_threshold=30, return_list=True
                    )
                    #  计算不同阈值下的 AUC ，分别表示误差在不同范围内的 AUC 值。
                    auc_thresholds = [30, 10, 5, 3]
                    for auc_threshold in auc_thresholds:
                        predictions[f"Auc_{auc_threshold}"] = torch.cumsum(
                            normalized_histogram[:auc_threshold], dim=0
                        ).mean() # 计算前 auc_threshold 个误差值的累计均值，得到 AUC 指标。

                    # 场景级 AUC
                    AUC_scene_dict[batch["seq_name"][0]] = torch.cumsum(normalized_histogram[:10], dim=0).mean()
                    # batch["seq_name"][0] 表示当前批次的场景名称。
                    # 计算该场景下 Auc_10，并存入 AUC_scene_dict，用于后续分析不同场景的误差表现。

                    # not pair-wise 即不考虑相机成对匹配关系，仅计算全局误差 计算非成对相机的相对旋转误差和相对平移误差
                    pair_rangle_deg, pair_tangle_deg = camera_to_rel_deg_pair(
                        pred_cameras, gt_cameras, accelerator.device, batch_size
                    )
                # 计算非成对相机 AUC ，@30°。
                    pair_Auc_30, pair_normalized_histogram = calculate_auc(
                        pair_rangle_deg, pair_tangle_deg, max_threshold=30, return_list=True
                    )
                # 非成对 AUC@10° / 3
                    predictions[f"PairAuc_10"] = torch.cumsum(pair_normalized_histogram[:10], dim=0).mean() # 非成对相机误差的归一化分布
                    predictions[f"PairAuc_3"] = torch.cumsum(pair_normalized_histogram[:3], dim=0).mean()

            if "pred_tracks" in predictions: # 这里检查 predictions 是否包含 pred_tracks（即预测的轨迹）
                pred_tracks = predictions["pred_tracks"]

                pred_tracks = pred_tracks[cfg.trackl] # cfg.trackl 可能是一个整数或索引数组，表示要提取的轨迹索引。

                pred_vis = predictions["pred_vis"] # pred_vis 代表模型预测的轨迹点可见性（可见性可能是 0~1 之间的概率值）

                # if cfg.debug:
                #     tracks = tracks * 2
                #     pred_tracks = pred_tracks * 2

                # if torch.isnan(pred_tracks).any() or torch.isinf(pred_tracks).any():
                #     for _ in range(100):
                #         print("Predicting NaN!!!!")
                # print(batch["seq_name"])

                track_dis = (tracks - pred_tracks) ** 2 # 计算gt 与pred 轨迹点之间的欧几里得距离。
                track_dis = torch.sqrt(torch.sum(track_dis, dim=-1)) # 计算坐标误差的平方：

                # if torch.isnan(track_dis).any() or torch.isinf(track_dis).any():
                #     track_dis_tmp = track_dis.sum(dim=-1).sum(dim=-1)
                #     print_idx = torch.logical_or(torch.isnan(track_dis_tmp), torch.isinf(track_dis_tmp))
                #     print_idx = torch.nonzero(print_idx)
                #     for _ in range(100):
                #         print("Meet NaN here!!!!")
                #         print(batch["seq_name"][print_idx])

                predictions["Track_m"] = track_dis.mean() # 计算所有轨迹误差的均值。
                predictions["Track_mvis"] = track_dis[tracks_visibility].mean() # 仅对可见轨迹计算误差均值，
                # 其中 tracks_visibility 是布尔掩码（True 代表该时间步的轨迹点是可见的）
                # TODO: need to check here again

                # import pdb;pdb.set_trace()

                # 去掉第一个时间步，避免起始误差影响。 去掉第一个时间步，获取可见轨迹点的掩码  只对可见轨迹点进行筛选
                # tracks_visibility[:, 1:] 是一个 布尔索引（mask），用于筛选 track_dis[:, 1:] 中 可见的轨迹点。
                predictions["Track_a1"] = (track_dis[:, 1:][tracks_visibility[:, 1:]] < 1).float().mean()

                if check_ni(predictions["Track_a1"]): # 检查 Track_a1 是否是 NaN
                    # TODO: something is wrong with Megadepth dataset loader
                    print("track a1 is NaN")
                    print(track_dis.shape)
                    print("dis:")
                    print(track_dis[:, 1:])
                    print("visi:")
                    print(tracks_visibility[:, 1:])
                    # print(batch['seq_id'])
                    # np.save("/data/home/jianyuan/src/ReconstructionJ/pose/pose_diffusion/tmp/debugNaN.npy", images.detach().cpu().numpy())

                pred_vis_binary = pred_vis > 0.5 # 果 pred_vis 大于 0.5，则认为该点是可见的，否则认为是不可见的
                correct_predictions = (pred_vis_binary == tracks_visibility).float()
                # 逐元素比较 pred_vis_binary 和 tracks_visibility，返回布尔值（True=预测正确，False=预测错误
                predictions["Vis_acc"] = correct_predictions.sum() / correct_predictions.numel()
                # 计算正确预测的数量，并除以总数，得到可见性预测的准确率

            if "pred_points" in predictions: #  计算 Chamfer 距离 Chamfer 距离是一种衡量两个点云相似度的度量方式，计算每个点到最近点的距离的均值。
                pred_points = predictions["pred_points"]
                chamfer = chamfer_distance(points, pred_points)[0]
                predictions["chamfer"] = chamfer

        ################################################################################################################
        if cfg.visualize:
            # TODO: clean here
            viz = vis_utils.get_visdom_connection(
                server=f"http://{cfg.viz_ip}", port=int(os.environ.get("VISDOM_PORT", 10088))
            )
            # viz = Visdom(server=f"10.200.160.58", port=10089)

            # import pdb;pdb.set_trace()
            # Visualize GT Cameras
            if gt_cameras is not None:
                # pred_cameras
                # gt_cameras

                first_seq_idx = torch.arange(frame_size)
                # pred_cameras_aligned = corresponding_cameras_alignment(cameras_src=pred_cameras[first_seq_idx], cameras_tgt=gt_cameras[first_seq_idx], estimate_scale=True, mode="extrinsics", eps=1e-9)
                cams_show = {"gt_cameras": gt_cameras[first_seq_idx]}

                first_seq_name = batch["seq_name"][0]

                if "pred_cameras" in predictions:
                    from pytorch3d.ops import corresponding_cameras_alignment
                    pred_cameras = predictions["pred_cameras"]
                    pred_cameras_aligned = corresponding_cameras_alignment(cameras_src=pred_cameras, cameras_tgt=gt_cameras, estimate_scale=True, mode="centers", eps=1e-9)

                    cams_show["pred_cameras"] = pred_cameras[first_seq_idx]
                    cams_show["pred_cameras_aligned"] = pred_cameras_aligned[first_seq_idx]
                    
                    
                fig = plot_scene({f"{first_seq_name}": cams_show})
                env_name = f"visual_{cfg.exp_name}_{step}"
                viz.plotlyplot(fig, env=env_name, win="cams")
                viz.images((images[0]).clamp(0, 1), env=env_name, win="img")
                # viz.images((images[0]).clamp(0, 1), env=env_name, win="img")

                # log_to_filename
                # viz_new =  Visdom(server=f"10.200.160.58", port=10089)

                # 10.200.160.58
                # viz_new.plotlyplot(fig, env="heyhey", win="cams")
                # viz_new.images((images[0]).clamp(0, 1), env="heyhey", win="img")
                print(env_name)
                import pdb;pdb.set_trace()

            # Visualize GT Tracks
            if "pred_tracks" in predictions:
                # if True:
                save_dir = f"{cfg.exp_dir}/visual_track" # 好像所有结果都保存在这个目录下了

                n_points = np.random.randint(20, 30)

                res_combined, _ = visualize_track(
                    predictions, images, tracks, tracks_visibility, cfg, step, viz, n_points=n_points, save_dir=save_dir,
                    visual_gt=True
                )

                save_track_visual = False
                if save_track_visual:
                    save_visual_track(res_combined, save_dir + ".png")

                import pdb;pdb.set_trace()


            if False:
                # camera_dict = {"pred_cameras": {}, "gt_cameras": {}}

                # for visidx in range(frame_size):
                # frame_size = 2
                # for visidx in range(frame_size):
                #     camera_dict["pred_cameras"][visidx] = pred_cameras[visidx]
                #     camera_dict["gt_cameras"][visidx] = gt_cameras[visidx]

                # # pcl = Pointclouds(points=points[0:1])
                # # camera_dict["points"] = {"pc": pcl}

                # fig = plotly_scene_visualization(camera_dict, frame_size)

                # viz.plotlyplot(fig, env="comeon", win="cams")
                # import pdb;pdb.set_trace()

                ########################################################
                # if "pred_points" in predictions:
                pcl = Pointclouds(points=points[0:1])
                # pred_points = predictions["pred_points"]
                # pred_pcl = Pointclouds(points=pred_points[0:1])
                # camera_dict["points"] = {"pc": pcl}

                pred_cameras_aligned = corresponding_cameras_alignment(
                    cameras_src=pred_cameras, cameras_tgt=gt_cameras, estimate_scale=True, mode="extrinsics", eps=1e-9
                )
                combined_dict = {
                    "scenes": {
                        "pred_cameras": pred_cameras[torch.range(0, frame_size - 1).long()],
                        "pred_cameras_aligned": pred_cameras_aligned[torch.range(0, frame_size - 1).long()],
                        "gt_cameras": gt_cameras[torch.range(0, frame_size - 1).long()],
                        "points": pcl,
                    }
                }

                fig = plot_scene(combined_dict)
                ########################################################
                viz.plotlyplot(fig, env=f"Perfect1", win="cams")

                show_img = view_color_coded_images_for_visdom(images[0])
                # viz.images(show_img.clamp(0,1), env=cfg.exp_name, win="imgs")

                viz.images(show_img.clamp(0, 1), env=f"Perfect1", win="imgs")
                # rel_rangle_deg.mean()
                # rel_tangle_deg
                import pdb

                pdb.set_trace()
                m = 1

            # viz.images(images[0], env=cfg.exp_name, win="imgs")
            # batch['seq_id']
            # cfg.train.normalize_cameras
        ################################################################################################################

        if training:
            stats.update(predictions, time_start=time_start, stat_set="train")
            if step % cfg.train.print_interval == 0:
                accelerator.print(stats.get_status_string(stat_set="train", max_it=max_it))
        else:
            stats.update(predictions, time_start=time_start, stat_set="eval")
            if step % cfg.train.print_interval == 0:
                accelerator.print(stats.get_status_string(stat_set="eval", max_it=max_it))

        if training:
            optimizer.zero_grad()
            accelerator.backward(loss)

            # if cfg.debug:
            #     for name, param in model.named_parameters():
            #         if "backbone" not in name:
            #             if param.grad is None:
            #                 print(f"Parameter '{name}' is unused.")
            #             else:
            #                 # Optionally check for zero gradients
            #                 # Note: This might not be necessary for just checking unused parameters
            #                 if torch.all(param.grad == 0):
            #                     print(f"Parameter '{name}' has zero gradient.")

            if cfg.train.clip_grad > 0:
                accelerator.clip_grad_norm_(model.parameters(), cfg.train.clip_grad)
                # total_norm_before_clipping = accelerator.clip_grad_norm_(model.parameters(), cfg.train.clip_grad)
                # print(total_norm_before_clipping)

            optimizer.step()
            lr_scheduler.step()

    if cfg.debug:
        import pdb

        pdb.set_trace()
        # AUC_scene_dict
        m = 1

    return True