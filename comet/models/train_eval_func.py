import os
import time
import psutil, tracemalloc, gc
import time

from pytorch3d.transforms import quaternion_to_matrix
from torch.cuda.amp import autocast
from tqdm import tqdm

from minipytorch3d.cameras import get_world_to_view_transform
from minipytorch3d.transform3d import Transform3d
from train_util import check_ni, record_and_print_cpu_memory_and_usage, process_spark_data
from pytorch3d.implicitron.tools import vis_utils
from pytorch3d.vis.plotly_vis import plot_scene
from metric import camera_to_rel_deg, calculate_auc, camera_to_rel_deg2
from lightglue import SuperPoint, SIFT, ALIKED

import psutil
import tracemalloc
import gc
import torch
from datetime import datetime

#
# class MemoryMonitor:
#     def __init__(self):
#         self.process = psutil.Process(os.getpid())
#         self.initial_memory = self.process.memory_info().rss
#         self.peak_memory = self.initial_memory
#         self.history = []
#         self.start_time = datetime.utcnow()
#
#     def get_memory_usage(self):
#         """获取当前内存使用情况"""
#         memory_info = self.process.memory_info()
#         virtual_memory = psutil.virtual_memory()
#
#         return {
#             'rss': memory_info.rss,  # 物理内存使用
#             'vms': memory_info.vms,  # 虚拟内存使用
#             'shared': memory_info.shared,  # 共享内存
#             'system_total': virtual_memory.total,  # 系统总内存
#             'system_available': virtual_memory.available,  # 系统可用内存
#             'system_percent': virtual_memory.percent  # 系统内存使用百分比
#         }
#
#     def log(self, step, location):
#         """记录内存使用"""
#         current = self.get_memory_usage()
#         self.peak_memory = max(self.peak_memory, current['rss'])
#
#         memory_info = {
#             'step': step,
#             'location': location,
#             'time': datetime.utcnow(),
#             'elapsed_time': (datetime.utcnow() - self.start_time).total_seconds(),
#             'memory': current
#         }
#
#         self.history.append(memory_info)
#
#         # 打印详细信息
#         print(f"\n📊 Memory Status at {location} (Step {step}):")
#         print(f"Physical Memory (RSS): {current['rss'] / 1024 ** 3:.2f}GB")
#         print(f"Virtual Memory (VMS): {current['vms'] / 1024 ** 3:.2f}GB")
#         print(f"Shared Memory: {current['shared'] / 1024 ** 3:.2f}GB")
#         print(f"Memory Growth: {(current['rss'] - self.initial_memory) / 1024 ** 3:+.2f}GB")
#         print(f"Peak Memory: {self.peak_memory / 1024 ** 3:.2f}GB")
#         print(f"System Memory Usage: {current['system_percent']}%")
#
#         # 检查内存增长
#         if current['rss'] > self.initial_memory * 1.5:  # 如果增长超过50%
#             print("\n⚠️ Warning: Significant memory growth detected!")
#             self.analyze_memory_usage()
#
#     def analyze_memory_usage(self):
#         """分析内存使用情况"""
#         print("\n🔍 Memory Analysis:")
#
#         # 获取大型对象
#         large_objects = []
#         for obj in gc.get_objects():
#             try:
#                 size = sys.getsizeof(obj)
#                 if size > 1024 * 1024:  # 大于1MB的对象
#                     large_objects.append((type(obj), size))
#             except:
#                 continue
#
#         # 打印大型对象信息
#         if large_objects:
#             print("\nLarge objects in memory:")
#             for obj_type, size in sorted(large_objects, key=lambda x: x[1], reverse=True)[:10]:
#                 print(f"{obj_type.__name__}: {size / 1024 ** 2:.2f}MB")
#
#         # 分析内存增长趋势
#         if len(self.history) > 1:
#             times = [h['elapsed_time'] for h in self.history]
#             memories = [h['memory']['rss'] for h in self.history]
#
#             if len(times) > 1:
#                 growth_rate = (memories[-1] - memories[0]) / (times[-1] - times[0])
#                 print(f"\nMemory growth rate: {growth_rate / 1024 ** 3:.3f}GB/s")


# 使用方式
# memory_monitor = MemoryMonitor() retain_graph=True





class QuaternionCameras:
    """
    Stores camera parameters using quaternion for rotation, without converting to rotation matrices.
    Mimics PerspectiveCameras-style interface.
    """

    def __init__(self, R, T, focal_length=1.0, principal_point=None, device="cpu"):
        self.device = device
        self.R = R.to(device)  # (N, 4) w x y z
        self.T = T.to(device)  # (N, 3)

        N = self.R.shape[0]

        # Format focal_length to (N, 2)
        if isinstance(focal_length, (float, int)):
            self.focal_length = torch.full((N, 2), focal_length, device=device)
        elif isinstance(focal_length, torch.Tensor):
            focal_length = focal_length.to(device)
            if focal_length.dim() == 0:
                self.focal_length = focal_length.expand(N, 2)
            elif focal_length.dim() == 1:
                self.focal_length = focal_length.view(-1, 1).expand(-1, 2)
            elif focal_length.dim() == 2:
                self.focal_length = focal_length
            else:
                raise ValueError("focal_length shape not recognized")
        else:
            raise TypeError("focal_length must be float, int or torch.Tensor")

        # Format principal_point to (N, 2)
        if principal_point is None:
            self.principal_point = torch.zeros((N, 2), device=device)
        else:
            # 1. 使用 as_tensor 保留原数据类型/设备，避免不必要的拷贝
            principal_point = torch.as_tensor(principal_point, dtype=torch.float32, device=device)
            if principal_point.dim() == 1:
                self.principal_point = principal_point.expand(N, 2)
            elif principal_point.dim() == 2:
                self.principal_point = principal_point
            else:
                raise ValueError("principal_point shape not recognized")

    def __repr__(self):
        return (
            f"QuaternionCameras(batch={self.R.shape[0]}, device={self.device})\n"
            f"  q: {self.R.shape}, T: {self.T.shape}\n"
            f"  focal_length: {self.focal_length.shape}, principal_point: {self.principal_point.shape}"
        )

    def get_world_to_view_transform(self) -> Transform3d:
        """
        Converts quaternion rotation and translation to a world-to-view Transform3d.
        """
        R_matrix = quaternion_to_matrix(self.R)  # (N, 3, 3)
        self.R_matrix=R_matrix
        T_vector = self.T  # (N, 3)
        return get_world_to_view_transform(R=R_matrix, T=T_vector)

def train_or_eval_fn(
    model, dataloader, cfg, optimizer, stats, accelerator, lr_scheduler, training=True, epoch=-1
):

    if training: # 根据 training 参数，判断是否进入训练模式。
        model.train()
    else:
        model.eval() #

    time_start = time.time()
    max_it = len(dataloader) # 每个 epoch 中的总步数（batch 数量）

    if cfg.track_by_spsg: # 这个条件判断是否使用 SuperPoint 或其他特征提取器来处理关键点。cfg.track_by_spsg 是一个配置项，决定是否启用这些特征提取器。
        # 这些行导入了一些特征提取器模块
        # —— 与推理阶段保持一致，强制输出固定数量的关键点 ——
        sp = SuperPoint(
               max_num_keypoints=cfg.train.track_num,
               detection_threshold=0.005).cuda().eval()
        sift = SIFT(max_num_keypoints=cfg.train.track_num).cuda().eval()

    AUC_scene_dict = {} # 初始化一个空字典，通常用于记录与 AUC 相关的场景数据。AUC（Area Under Curve）


    for step, batch in enumerate(tqdm(dataloader)): # 进入数据加载循环，逐步处理 dataloader 中的每个批次（batch）。
        # log_mem(step)
        # print(batch["seq_name"])
        # memory_monitor.log(step, "Batch Start")
        if step == 100: #X 当迭代到第 100 个批次时，记录并打印当前的 CPU 内存使用情况。这通常用于调试。
            record_and_print_cpu_memory_and_usage()

        gt_cameras = None # 初始化 gt_cameras 为 None，用于存储地面真实值的相机信息。


        #############先进行数据处理########
        (
            images,
            translation,
            rotation,
            fl,
            pp
        ) = process_spark_data(batch, accelerator.device, cfg)
        # 我们假设 process_spark_data 返回的 images 形状是 [B, S, C, H, W]；translation、rotation、fl、pp 分别是 [B,S,3]、[B,S,3,3]、[B,S,2]、[B,S,2]
        # 调用 process_co3d_data 函数来处理当前批次的batch，是一个字典。这个函数返回一组图像和其他相关数据，包括图像、平移、旋转、焦距、主点坐标
        # memory_monitor.log(step, "After Data Processing")


        ######### 通过合并不同的特征提取器的输出并筛选出数量最合适的关键点，作为初始的的轨迹跟踪点。##########
        if cfg.track_by_spsg and (not cfg.labor_input_traj):
            # 判断是否启用了 track_by_spsg（即启用了特征提取器），并且当前不是推理阶段（cfg.inference 为 False）。
            # use keypoints as the starting points of tracks
            images_for_kp = images[:, 0] # 将图像数据赋值给 images_for_kp，这是用于提取关键点的数据
            bbb,ttt ,_, _, _ = images.shape # bbb 是批次大小，nnn 是轨迹数量，ppp 是每个轨迹的维度。

            # —— 调用 extract() 接口，保证返回 [B, track_num, 2] 张量 ——
            kp0_sp_list = []
            for i in range(bbb):
                img_i = images_for_kp[i] # [1, C, H, W]
                out_i = sp.extract(img_i, invalid_mask=None)
                kp0_sp_list.append(out_i["keypoints"].squeeze(0))  # [N_sp, 2]

            kp0_sift_list = []
            for i in range(bbb):
                img_i = images_for_kp[i]  # [1, C, H, W]
                out_i = sift.extract(img_i, invalid_mask=None)
                kp0_sift_list.append(out_i["keypoints"].squeeze(0))

            # 1) 合并每个样本的 SP 与 SIFT 特征点，得到 kp0_list
            kp0_list = [
                torch.cat([sp_pts, sift_pts], dim=0)  # shape=[Ni_sp + Ni_sift, 2]
                for sp_pts, sift_pts in zip(kp0_sp_list, kp0_sift_list)
            ]

            # 2) 找出所有样本中最小的特征点数 min_n
            min_n = min(p.shape[0] for p in kp0_list)

            # 3) 确定最终要选的点数 T = min(min_n, cfg.train.track_num)
            T = min(min_n, cfg.train.track_num)

            # 4) 对每个样本随机抽 T 个点
            kp0_selected = []
            for p in kp0_list:
                # 随机打乱并取前 T
                idx = torch.randperm(p.shape[0], device=p.device)[:T]
                kp0_selected.append(p[idx])

            # 5) 堆叠成 [B, T, 2]
            kp0 = torch.stack(kp0_selected, dim=0)  # [B, T, 2]

            new_track_num = kp0.shape[-2] #从获取合并后的关键点数量。
            if new_track_num > cfg.train.track_num: # 如果关键点数量超过了预设的最大数量 512，则随机选择一部分关键点。
                indices = torch.randperm(new_track_num)[: cfg.train.track_num] # 随机选择前 cfg.train.track_num 个关键点。
                kp0 = kp0[:, indices, :] # 将 kp0 限制为选择的关键点。

            # 假设 kp0 [B, N, 2]
            kp0 = kp0.unsqueeze(1).expand(bbb, ttt, -1, -1)  # [B, ttt, N,2]
            tracks = kp0
            tracks_visibility = torch.ones(bbb, ttt, kp0.shape[-2], device=accelerator.device, dtype=torch.bool)
        else:
            tracks = None
            tracks_visibility = None

        if rotation is not None: # 相机和数据处理：
            # 如果 rotation（旋转矩阵）不是 None，即模型的旋转矩阵存在，那么我们将根据提供的旋转矩阵、平移向量以及焦距、主点等信息来创建gt相机
            gt_cameras = QuaternionCameras(
                focal_length=fl.reshape(-1, 2), #焦距
                principal_point=pp.reshape(-1, 2),# 主点
                R=rotation.reshape(-1, 4),
                T=translation.reshape(-1, 3),
                device=accelerator.device,
            )

        if training: # 进入训练模式。如果是训练，我们使用模型进行前向传播。
            predictions = model(
                images,
                gt_cameras=gt_cameras,
                training=True,
                tracks=tracks,
                tracks_visibility=tracks_visibility,
            )
            predictions["loss"] = predictions["loss"].mean()
            loss = predictions["loss"]
        else:
            with torch.no_grad():
                # 如果评估模式，则调用模型进行前向传播。使用 torch.no_grad() 来禁止梯度计算，因为在评估模式下我们不需要反向传播。
                predictions = model(
                    images,
                    gt_cameras=gt_cameras,
                    training=False,
                    tracks=tracks,
                    tracks_visibility=tracks_visibility,
                )
            predictions["loss"] = predictions["loss"].mean()

        # memory_monitor.log(step, "After Forward Pass")

        # Computing Metrics 用于 计算评估指标 评估预测的相机姿态 (pred_cameras) 与真实相机姿态 (gt_cameras) 之间的误差。
        with torch.no_grad(): # 使用 torch.no_grad() 来减少内存占用并加速计算。
            if "pred_cameras" in predictions: # 检查 predictions 字典中是否包含 pred_cameras（即模型是否预测了相机参数）
                with autocast(dtype=torch.double):
                    # 使用 autocast(dtype=torch.double) 以 double (64-bit) 精度进行计算，提高计算精度，特别是在涉及旋转矩阵和角度计算时
                    ###################都传入绝对姿态 而且是类的形式 进行比较 总共286个结果 是两两之间视角的误差（him and me)###############
                    pred_cameras = predictions["pred_cameras"]  # 绝对
                    rel_rangle_deg_him, rel_tangle_deg_him = camera_to_rel_deg(
                        pred_cameras, gt_cameras, accelerator.device, bbb
                    )  # 计算相对旋转角误差 (rel_rangle_deg) 和相对平移角误差 将旋转矩阵转换为角度误差，以及 计算平移向量的角度误差
                    ###################都传入绝对姿态 而且是类的形式 进行比较 总共286个结果 是两两之间视角的误差###############
                    pred_cameras = predictions["pred_pose_enc"]
                    gt_cameras = predictions["gt_pose_enc"]
                    rel_rangle_deg_rel2one, rel_tangle_deg_rel2one, R_avg, error_euler, _ = camera_to_rel_deg2(
                        pred_cameras, gt_cameras, accelerator.device, bbb
                    )  # 计算相对旋转角误差 (rel_rangle_deg) 和相对平移角误差 将旋转矩阵转换为角度误差，以及 计算平移向量的角度误差
                    ###################都传入相对姿态 而且是类的形式 进行比较 总共24个结果 是与第一帧之间相对视角的误差###############

                    predictions["X_err"] = error_euler[2]
                    predictions["Y_err"] = error_euler[1]
                    predictions["Z_err"] = error_euler[0]
                    predictions["R_avg"] = R_avg
                    # metrics to report
                    thresholds = [5, 10, 15]
                    # 计算准确率
                    # 1 是 每两帧之间都计算一次误差 但是他是左乘 我就觉得和我们之前的那个不一样
                    # 2 是 每两帧之间都计算一次误差 右乘
                    # 3 是 只和第一帧 计算相对误差 它的平移 是直接相减得到的 但前面的相对平移是在第一帧的坐标系下的值
                    for threshold in thresholds:
                        predictions[f"Racc_him_{threshold}"] = (
                                rel_rangle_deg_him < threshold).float().mean()  # 计算 Racc_5, Racc_15, Racc_30（旋转准确率）
                        predictions[f"Tacc_him_{threshold}"] = (
                                rel_tangle_deg_him < threshold).float().mean()  # # 结果: (1+0+1+0+0)/5 = 0.4
                    # 计算 AUC（累积误差分布）  累计误差曲线 (AUC, Area Under Curve)，用来衡量模型的整体表现。
                    Auc_30, normalized_histogram = calculate_auc(
                        rel_rangle_deg_him, rel_tangle_deg_him, max_threshold=30, return_list=True
                    )
                    #  计算不同阈值下的 AUC ，分别表示误差在不同范围内的 AUC 值。
                    auc_thresholds = [30, 10, 5, 3]
                    for auc_threshold in auc_thresholds:
                        predictions[f"Auc_{auc_threshold}"] = torch.cumsum(
                            normalized_histogram[:auc_threshold], dim=0
                        ).mean() # 计算前 auc_threshold 个误差值的累计均值，得到 AUC 指标。

                    # 场景级 AUC
                    scene_name = batch["seq_name"][0]
                    # 先把当前场景的 AUC 记录到字典（可选）
                    AUC_scene_dict[scene_name] = torch.cumsum(normalized_histogram[:10], dim=0).mean()
                    # 然后把它写回 predictions
                    predictions[f"Auc_scene_{scene_name}"] = AUC_scene_dict[scene_name]
                    # batch["seq_name"][0] 表示当前批次的场景名称。
                    # 计算该场景下 Auc_10，并存入 AUC_scene_dict，用于后续分析不同场景的误差表现。


        if training:
            stats.update(predictions, time_start=time_start, stat_set="train")
            # ，调用 stats.update() 更新统计信息，其中传入当前的 predictions、开始时间 time_start 以及统计集合名称 stat_set 设置为 "train"
            if step % cfg.train.print_interval == 0:
                accelerator.print(stats.get_status_string(stat_set="train", max_it=max_it))
                accelerator.print(
                    f"  Batch Loss Trace:"
                    f"\n    loss_trans   : {predictions.get('loss_trans', 0):.6f}"
                    f"   loss_rot     : {predictions.get('loss_rot', 0):.6f}"
                )
        else:
            stats.update(predictions, time_start=time_start, stat_set="eval")
            if step % cfg.train.eval_print_interval == 0:
                accelerator.print(stats.get_status_string(stat_set="eval", max_it=max_it))
                accelerator.print(
                    f"  Batch Loss Trace:"
                    f"\n    loss_trans   : {predictions.get('loss_trans', 0):.6f}"
                    f"   loss_rot     : {predictions.get('loss_rot', 0):.6f}"
                    f"   loss_motions : {predictions.get('loss_motions', 0):.6f}"
                )

        if training:
            optimizer.zero_grad() #清空梯度
            with torch.autograd.detect_anomaly():
                accelerator.backward(loss)

            # accelerator.backward(loss) # 反向传播
            # memory_monitor.log(step, "After Backward Pass")

            # for name, p in model.named_parameters():
            #     if p.grad is None: continue
            #     print(f"{name}: min={p.grad.min():.4e}, max={p.grad.max():.4e}, mean={p.grad.mean():.4e}")
            if cfg.train.clip_grad > 0:
                total_norm_before = accelerator.clip_grad_norm_(model.parameters(), cfg.train.clip_grad)
                # print(f"[GradClip] max_norm={cfg.train.clip_grad}, before={total_norm_before:.4f}")

            # 接下来执行 optimizer.step()，利用计算得到的梯度更新模型参数。
            # 同时，调用 lr_scheduler.step() 更新学习率，遵循预设的学习率调整策略
            optimizer.step()
            # memory_monitor.log(step, "After Optimizer Step")
            lr_scheduler.step()

            lr_scheduler.step()
            torch.cuda.empty_cache()

            # 清理中间变量
            del predictions
            del images
            del translation
            del rotation
            del fl
            del pp
            del tracks
            del tracks_visibility
            if gt_cameras is not None:
                del gt_cameras
            if training:
                del loss


            # break

    return True