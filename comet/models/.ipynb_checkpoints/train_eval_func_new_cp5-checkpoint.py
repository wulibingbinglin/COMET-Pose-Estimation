import json
import os
import time

import numpy as np
import torch.nn.functional as F

from matplotlib import pyplot as plt
from torch.cuda.amp import autocast
from tqdm import tqdm
from minipytorch3d.cameras import get_world_to_view_transform
from minipytorch3d.rotation_conversions import quaternion_to_matrix
from minipytorch3d.transform3d import Transform3d
from train_util import check_ni, record_and_print_cpu_memory_and_usage, process_spark_data, process_spark_data2
from metric import camera_to_rel_deg, calculate_auc, camera_to_rel_deg2, camera_to_rel_deg3
from lightglue import SuperPoint, SIFT
import torch
import torchvision.transforms.functional as TF

from visualizer import Visualizer

class QuaternionCameras:
    """
    Stores camera parameters using quaternion for rotation, without converting to rotation matrices.
    Mimics PerspectiveCameras-style interface.
    """

    def __init__(self, R, T_uvz, T, focal_length=1.0, principal_point=None,ratio=None, device="cpu"):
        self.device = device
        self.R = R.to(device)  # (N, 4) w x y z
        self.T = T.to(device)  # (N, 3)
        self.T_uvz = T_uvz.to(device)
        self.ratio = ratio

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
        self.R_matrix = R_matrix
        T_vector = self.T # (N, 3)
        return get_world_to_view_transform(R=R_matrix, T=T_vector)


class TrainingMonitor:
    def __init__(self, save_dir, threshold=1000, window_size=50, max_checkpoints=5, epoch=-1):
        self.save_dir = save_dir
        self.threshold = threshold
        self.window_size = window_size
        self.max_checkpoints = max_checkpoints
        self.loss_history = []
        self.normal_checkpoints = []
        self.epoch = epoch

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'normal_checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'anomaly_checkpoints'), exist_ok=True)

        # 创建日志文件
        self.log_file = os.path.join(save_dir, f'training_log_{time.strftime("%Y%m%d_%H%M%S")}.txt')

    def log_message(self, message):
        """写入日志"""
        with open(self.log_file, 'a') as f:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"[{timestamp}] {message}\n")

    def check_anomaly(self, current_loss):
        self.loss_history.append(current_loss)
        if len(self.loss_history) > self.window_size:
            self.loss_history.pop(0)

        if current_loss > self.threshold:
            return True

        if len(self.loss_history) >= 2 and \
                self.loss_history[-1] > self.loss_history[-2] * 100:
            return True

        return False

    def save_checkpoint(self, step, model, batch, predictions, is_anomaly=False):
        """保存检查点"""
        checkpoint_dir = 'anomaly_checkpoints' if is_anomaly else 'normal_checkpoints'
        checkpoint_path = os.path.join(
            self.save_dir,
            checkpoint_dir,
            f'epoch_{self.epoch}_step_{step:06}.pth'
        )

        # 保存检查点
        checkpoint_data = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'batch_data': batch,
            'predictions': predictions,
            'loss_history': self.loss_history.copy(),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        torch.save(checkpoint_data, checkpoint_path)

        # 如果是正常检查点，维护检查点历史
        if not is_anomaly:
            self.normal_checkpoints.append(checkpoint_path)
            if len(self.normal_checkpoints) > self.max_checkpoints:
                # 删除最老的检查点
                old_checkpoint = self.normal_checkpoints.pop(0)
                if os.path.exists(old_checkpoint):
                    os.remove(old_checkpoint)

        return checkpoint_path

    def update(self, step, model, batch, predictions, loss):
        """更新监控状态"""
        current_loss = loss.item()

        # 检查是否异常
        is_anomaly = self.check_anomaly(current_loss)

        if step % 20 == 0:
            # 记录详细信息
            self.log_message(
                f"普通输出-------Step: {step}------\n"
                f"Loss: {current_loss} "
                f"Loss Trans: {predictions.get('loss_trans', 0)} "
                f"Loss Rot: {predictions.get('loss_rot', 0)}\n"
                f"seq_name: {batch['seq_name']}"
                f"image_names: {batch['image_names']}"
            )
        else:
            self.log_message(
                f"普通输出-------Step: {step}------\n"
                f"Loss: {current_loss} "
                f"Loss Trans: {predictions.get('loss_trans', 0)} "
                f"Loss Rot: {predictions.get('loss_rot', 0)}\n"
            )

        # 如果发现异常
        if is_anomaly:
            # 使用更醒目的分隔符和格式化的时间戳
            self.log_message("!" * 25+ f"警告：在epoch{self.epoch}_step {step}检测到异常!"+ "!" * 25)
            self.log_message(f"当前batch信息："
                             f"Images shape: {batch['images'].shape}  "
                             f"Loss值: {current_loss}\n"
                             f"seq_name: {batch['seq_name']}"
                             f"image_names: {batch['image_names']}")



def save_track_flow(img_t, predictions, sel_first_name, cfg, visual_dir=None, fps=10):
    """
    images: list[PIL.Image] or ndarray (T,H,W,3) or tensor (T,3,H,W) or list of numpy arrays
    predictions: dict-like containing key "pred_tracks" -> (T,N,2) or (B,T,N,2)
    sel_first_name: str, unique id for this sequence (no extension)
    cfg: config object, must have cfg.visual_track boolean
    visual_dir: if None, use os.getcwd()/track_visual
    """
    if not cfg.visual_track:
        return None

    # --- prepare save dir & filename
    if visual_dir is None:
        visual_dir = os.path.join(os.getcwd(), "track_visual")
    os.makedirs(visual_dir, exist_ok=True)

    # sanitize filename
    filename = sel_first_name
    tracks = predictions["pred_tracks"]
    if tracks is None:
        raise ValueError("predictions must contain 'pred_tracks'")
    # normalize dims:
    if tracks.ndim == 3:
        # (T,N,2) -> (1,T,N,2)
        tracks = tracks.unsqueeze(0)
    elif tracks.ndim == 4:
        # assume (B,T,N,2)
        pass
    else:
        raise ValueError("pred_tracks shape not supported")

    # ensure dtype float and on cpu
    tracks_t = tracks.float().cpu()

    # --- visibility (optional)
    visibility = None

    # --- instantiate Visualizer and visualize
    vis = Visualizer(save_dir=visual_dir, linewidth=1, fps=fps)
    un_norm = True
    if un_norm:
        device = img_t.device  # 获取 img_t 当前所在设备
        mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]

        video_denorm = img_t * std + mean

    # 2. 缩放到[0,255]范围并转换为uint8
    video_input = (video_denorm * 255.0).clamp(0, 255).byte()

    filename = filename.replace('\\', '_').replace('/', '_')

    # call visualize
    vis.visualize(video_input, tracks_t, visibility=visibility, filename=filename, save_video=True)

    # full saved path:
    saved_path = os.path.join(visual_dir, f"{filename}.mp4")
    print(f"Saved visualization to: {saved_path}")
    return saved_path


def sample_extra_points(mask, need_extra, device):
    """
    从mask里随机采样点
    mask: [H, W] bool
    """
    ys, xs = torch.where(mask)
    if ys.numel() == 0:
        return None
    idx = torch.randint(0, ys.shape[0], (need_extra,), device=device)
    return torch.stack([xs[idx], ys[idx]], dim=1).float().to(device)
# 2) 过滤并补点函数
def filter_and_pad(pts, mask0, min_pts, max_pts, sel_first_name):
    """
    pts: [N, 2] keypoints
    mask0: [H, W] bool, 第一帧的mask
    min_pts, max_pts: 需要的点数范围
    """
    H, W = mask0.shape
    device = pts.device

    xy = pts.clone()
    xy[:, 0] = xy[:, 0].round().clamp(0, W - 1)
    xy[:, 1] = xy[:, 1].round().clamp(0, H - 1)
    xs = xy[:, 0].long()
    ys = xy[:, 1].long()

    mask0 = mask0.to(device) 
    target_region = mask0.bool()  
    keep_idx = target_region[ys, xs]
    keep = pts[keep_idx]
    if keep.shape[0] < min_pts:
        need_extra = min_pts - keep.shape[0]

        extra_pts = sample_extra_points(target_region, need_extra, device)

        if extra_pts is None or extra_pts.shape[0] < need_extra:
            # dilate 一圈
            print("在目标外圆圈补,这个序列是：",sel_first_name)
            dilated = F.max_pool2d(target_region.float()[None, None], kernel_size=3, stride=1, padding=1)[0, 0].bool()
            dilated = dilated & (~target_region) 
            remain = need_extra if extra_pts is None else need_extra - extra_pts.shape[0]
            extra2 = sample_extra_points(dilated, remain, device)
            if extra2 is not None:
                extra_pts = extra2 if extra_pts is None else torch.cat([extra_pts, extra2], dim=0)

        if extra_pts is None or extra_pts.shape[0] < need_extra:
            print("在目标外随机补,这个序列是：",sel_first_name)
            ys_all, xs_all = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing="ij"
            )
            coords_all = torch.stack([xs_all.flatten(), ys_all.flatten()], dim=1).float()
            remain = need_extra if extra_pts is None else need_extra - extra_pts.shape[0]
            idx = torch.randint(0, coords_all.shape[0], (remain,), device=device)
            extra3 = coords_all[idx]
            extra_pts = extra3 if extra_pts is None else torch.cat([extra_pts, extra3], dim=0)

        keep = torch.cat([keep, extra_pts], dim=0)

    if keep.shape[0] > max_pts:
        idx = torch.randperm(keep.shape[0], device=device)[:max_pts]
        keep = keep[idx]

    return keep

import os
import cv2
import numpy as np
import torch

def quat_to_rotmat_np(q):
    q = np.asarray(q, dtype=np.float64)
    assert q.ndim == 2 and q.shape[1] == 4
    w, x, y, z = q[:,0], q[:,1], q[:,2], q[:,3]
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    R = np.zeros((q.shape[0], 3, 3), dtype=np.float64)
    R[:,0,0] = ww + xx - yy - zz
    R[:,0,1] = 2*(xy - wz)
    R[:,0,2] = 2*(xz + wy)
    R[:,1,0] = 2*(xy + wz)
    R[:,1,1] = ww - xx + yy - zz
    R[:,1,2] = 2*(yz - wx)
    R[:,2,0] = 2*(xz - wy)
    R[:,2,1] = 2*(yz + wx)
    R[:,2,2] = ww - xx - yy + zz
    return R

def ensure_images_list(images):
    """
    将 images (torch 或 numpy) 转为 list[np.uint8 HxWx3]。
    支持 [S,3,H,W], [B,S,3,H,W], [S,H,W,3]。
    """
    img_t = images.detach().cpu()[0]
    un_norm = True
    if un_norm:
        device = img_t.device  # 获取 img_t 当前所在设备
        mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]

        video_denorm = img_t * std + mean

    # 2. 缩放到[0,255]范围并转换为uint8
    arr = (video_denorm.permute(0,2,3,1).cpu().numpy() * 255.0).clip(0,255).astype(np.uint8)

    return [arr[i] for i in range(arr.shape[0])]  # 将张量转化为数组

def draw_dashed_line(img, pt1, pt2, color, thickness=1, gap=8):
    """在 img 上绘制虚线（pt1,pt2 为 (x,y)）"""
    x1,y1 = pt1; x2,y2 = pt2
    dist = int(np.hypot(x2-x1, y2-y1))
    if dist <= 2:
        cv2.line(img, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)
        return
    # number of segments
    dash_len = max(1, int(gap * 0.5))
    num = dist // dash_len
    if num <= 1:
        cv2.line(img, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)
        return
    for i in range(0, num, 2):
        start = i / num
        end = min((i+1) / num, 1.0)
        sx = int(round(x1 + (x2-x1) * start))
        sy = int(round(y1 + (y2-y1) * start))
        ex = int(round(x1 + (x2-x1) * end))
        ey = int(round(y1 + (y2-y1) * end))
        cv2.line(img, (sx,sy), (ex,ey), color, thickness, lineType=cv2.LINE_AA)


def save_first_k_pose_images(
    images_path,
    pred_cameras,
    gt_cameras,
    out_dir,
    sel_name="seq",
    K=None,
    axis_len=0.2,
    k=3,
    R_matrix_gt=None,
):
    """
    保存序列前 k 帧的姿态对比图（每帧一张图片）。
    - pred/gt 的 R 可以是 (N,4)(四元数 w,x,y,z)。   t是 n 3
    返回：saved_paths (list)
    """
    os.makedirs(out_dir, exist_ok=True)

    # 从路径加载图像
    frames = []
    for img_path in images_path:
        if isinstance(img_path, list):
            img_path = img_path[0] 
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告: 无法读取图像 {img_path}")
            # 创建一个空白图像作为占位符
            img = np.zeros((480, 640, 3), dtype=np.uint8)
        frames.append(img)

    n_save = k

    if K is None:
        print("K is none ")
        return
    K = np.asarray(K, dtype=np.float64)

    # extract T and R arrays from camera objects (to numpy)
    T_gt = np.asarray(gt_cameras.T.detach().cpu().numpy())  # (N,3)
    T_pred = np.asarray(pred_cameras.T.detach().cpu().numpy())  # (N,3)
    R_pred_raw = pred_cameras.R.detach().cpu().numpy()  # (N,4)
    R_pred_mats = quat_to_rotmat_np(R_pred_raw) # n 3 3

    # R_gt_raw = gt_cameras.R.detach().cpu().numpy()      # (N,4)
    # R_gt_mats = quat_to_rotmat_np(R_gt_raw)

    R_gt_mats = R_matrix_gt.detach().cpu().numpy()  # n 3 3

    # print("R_gt_mats",R_gt_mats)

    # Basic checks
    N = T_gt.shape[0]
    assert R_pred_mats.shape[0] >= n_save and R_gt_mats.shape[0] >= n_save and N >= n_save, \
        f"camera count ({N}) shorter than frames to save ({n_save})"

    # object points (origin + three axis endpoints)
    P_obj = np.array([
        [0.0, 0.0, 0.0],
        [axis_len, 0.0, 0.0],
        [0.0, axis_len, 0.0],
        [0.0, 0.0, axis_len],
    ], dtype=np.float64)

    colors_pred = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR
    colors_gt = [(128, 0, 255), (0, 128, 0), (128, 0, 0)]
    # 定义两种颜色方案（使用OpenCV的BGR格式）：
    # 预测坐标系颜色：蓝色(X)、绿色(Y)、红色(Z)
    # 真实坐标系颜色：紫色(X)、暗绿色(Y)、暗蓝色(Z)

    for i in range(n_save):
        img = frames[i].copy()

        t_gt = T_gt[i].reshape(3)  # use GT translation as anchor
        t_pred = T_pred[i].reshape(3)
        R_pred = R_pred_mats[i]
        R_gt = R_gt_mats[i]

        def project(R,T):
            pts_cam = (R @ P_obj.T).T + T
            uvw = (K @ pts_cam.T).T
            uv = uvw[:, :2] / uvw[:, 2:3]
            return uv

        uv_pred = project(R_pred,t_pred)
        uv_gt = project(R_gt,t_gt)

        origin_pred = tuple(np.round(uv_pred[0]).astype(int))
        origin_gt = tuple(np.round(uv_gt[0]).astype(int))

        # 在原点上绘制标记点
        cv2.circle(img, origin_pred, 7, (255, 255, 255), -1)  # 半径从 5 改 7
        cv2.circle(img, origin_gt, 7, (0, 0, 0), -1)  # 黑色圆点标记GT原点
        axis_name = ['x', 'y', 'z']
        # 画 pred (实线)
        for j in range(3):
            p = tuple(np.round(uv_pred[j + 1]).astype(int))
            cv2.line(img, origin_pred, p, colors_pred[j], 2, cv2.LINE_AA)
            cv2.putText(img, axis_name[j], p,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,  # 字体大小从 0.5 改 0.8
                        colors_pred[j],
                        2,  # 字体粗细从 1 改 2
                        cv2.LINE_AA)

        # 画 gt (虚线/淡色)
        for j in range(3):
            p = tuple(np.round(uv_gt[j + 1]).astype(int))
            draw_dashed_line(img, origin_gt, p, colors_gt[j], thickness=2, gap=6)
            cv2.putText(img, axis_name[j], p,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,  # 字体大小从 0.5 改 0.8
                        colors_pred[j],
                        2,  # 字体粗细从 1 改 2
                        cv2.LINE_AA)

        # 使用序列名称和帧号创建文件名
        filename = sel_name.replace('/', '_')

        # 从路径中提取帧号
        frame_name = os.path.basename(images_path[i])
        if isinstance(frame_name, list):
            frame_name = frame_name[0]
        frame_name = os.path.splitext(frame_name)[0]  # 去掉扩展名

        out_path = os.path.join(out_dir, f"{filename}_{frame_name}.png")
        try:
            cv2.imwrite(out_path, img)
        except Exception as e:
            print(f"保存时出错: {e}")
            print(f"保存失败: {out_path}")

def train_or_eval_fn(
        model, dataloader, cfg, optimizer, stats, accelerator, lr_scheduler, training=True, epoch=-1
):
    if training:  
        model.train()
    else:
        model.eval()  #
        # 增加一个标记，防止重复打印
        has_printed_speed = False

    time_start = time.time()
    max_it = len(dataloader)  # 每个 epoch 中的总步数（batch 数量）

    if cfg.track_by_spsg: 
        sp = SuperPoint(
            max_num_keypoints=cfg.train.track_num,
            detection_threshold=0.005).cuda().eval()
        sift = SIFT(max_num_keypoints=cfg.train.track_num).cuda().eval()

    AUC_scene_dict = {}  # 初始化一个空字典，通常用于记录与 AUC 相关的场景数据。AUC（Area Under Curve）
    x,y,z = 0,0,0
    for step, batch in enumerate(tqdm(dataloader)): 
        gt_cameras = None 
        (
            images,
            T_xyz,
            T_uvz,
            rotation,
            fl,
            pp,
            ratio,
            sel_first_name,
            image_names,
            mask,
            R_matrix_gt
        ) = process_spark_data2(batch, accelerator.device, cfg)

        bbb, ttt, _, _, _ = images.shape 
        # try:
        if cfg.enable_track and cfg.track_by_spsg and (not cfg.labor_input_traj):
            # use keypoints as the starting points of tracks
            images_for_kp = images[:, 0]  

            # —— 调用 extract() 接口，保证返回 [B, track_num, 2] 张量 ——
            kp0_sift_list = []
            kp0_sp_list = []
            for i in range(bbb):
                img_i = images_for_kp[i]  # [1, C, H, W]
                out_i1 = sp.extract(img_i)
                kp0_sp_list.append(out_i1["keypoints"].squeeze(0))  # [N_sp, 2]
                out_i2 = sift.extract(img_i)
                kp0_sift_list.append(out_i2["keypoints"].squeeze(0))

            kp0_list = [
                torch.cat([sp_pts, sift_pts], dim=0)  # shape=[Ni_sp + Ni_sift, 2]
                for sp_pts, sift_pts in zip(kp0_sp_list, kp0_sift_list)
            ]

            # 3) 对每个样本进行处理
            min_required = 256  # 最低点数
            max_allowed = cfg.train.track_num

            kp0_selected = []
            mask0 = mask.bool()  # [H, W]，取第一帧mask 1 256 256

            for i, pts in enumerate(kp0_list):
                filtered = filter_and_pad(pts, mask0[i], min_required, max_allowed, sel_first_name)
                kp0_selected.append(filtered)

            # 5) 堆叠成 [B, T, 2]
            kp0 = torch.stack(kp0_selected, dim=0)  # [B, T, 2]

            # print("数量",kp0.size(1),"个，sp",sp_counts,"个，sift",sift_counts,"个")


            # 假设 kp0 [B, N, 2]
            kp0 = kp0.unsqueeze(1).expand(bbb, ttt, -1, -1)  # [B, ttt, N,2]
            tracks = kp0
            tracks_visibility = torch.ones(bbb, ttt, kp0.shape[-2], device=accelerator.device, dtype=torch.bool)
        else:
            tracks = None
            tracks_visibility = None

        if rotation is not None: 
            gt_cameras = QuaternionCameras(
                focal_length=fl.reshape(-1, 2),  # 焦距
                principal_point=pp.reshape(-1, 2),  # 主点
                R=rotation.reshape(-1, 4),
                T_uvz=T_uvz.reshape(-1, 3),
                T=T_xyz.reshape(-1, 3),
                ratio=ratio,
                device=accelerator.device
            )

        if training:  # 进入训练模式。如果是训练，我们使用模型进行前向传播。 if step
            predictions = model(
                images,
                gt_cameras=gt_cameras,
                training=True,
                tracks=tracks,
                tracks_visibility=tracks_visibility,
            )
            predictions["loss"] = predictions["loss"].mean()
            loss = predictions["loss"]
            # print("11true_loss", loss)
        else:
            with torch.no_grad():
                predictions = model(
                    images,
                    gt_cameras=gt_cameras,
                    training=False,
                    tracks=tracks,
                    tracks_visibility=tracks_visibility,
                )
            predictions["loss"] = predictions["loss"].mean()

        if cfg.visual_track:
            save_track_flow(images, predictions, sel_first_name[0], cfg)
        with torch.no_grad():  # 使用 torch.no_grad() 来减少内存占用并加速计算。
            if "gt_pose_enc" in predictions:  # 检查 predictions 字典中是否包含 pred_cameras（即模型是否预测了相机参数）
                with autocast(dtype=torch.double):
                    pred_cameras = predictions["pred_cameras"]  # 绝对 且是xyz
                    rel_rangle_deg_him, rel_tangle_deg_him, T_avg, Tx_mse, Ty_mse, Tz_mse = camera_to_rel_deg3(
                        pred_cameras, gt_cameras, accelerator.device, bbb
                    ) 
                    pred_cameras_rel = predictions["pred_pose_enc"]
                    gt_cameras_rel = predictions["gt_pose_enc"]  # 变为相对的uvz 和
                    _, _, R_avg, error_euler,  acc_5_deg = camera_to_rel_deg2(
                        pred_cameras_rel, gt_cameras_rel, accelerator.device, bbb
                    )  
                    predictions["X_err"] = error_euler[2]
                    predictions["Y_err"] = error_euler[1]
                    predictions["Z_err"] = error_euler[0]
                    predictions["R_avg"] = R_avg
                    predictions["T_avg"] = T_avg
                    predictions["Tx_mse"] = Tx_mse
                    predictions["Ty_mse"] = Ty_mse
                    predictions["Tz_mse"] = Tz_mse
                    # print("R_avg, X_err","Y_err","Z_err:", R_avg, error_euler[2], error_euler[1], error_euler[0])
                    predictions["acc@5deg_x"] = acc_5_deg[2]
                    predictions["acc@5deg_y"] = acc_5_deg[1]
                    predictions["acc@5deg_z"] = acc_5_deg[0]
                    x+=acc_5_deg[2]
                    y+=acc_5_deg[1]
                    z+=acc_5_deg[0]
                    thresholds = [5, 10, 15]
                    for threshold in thresholds:
                        predictions[f"Racc_him_{threshold}"] = (
                                rel_rangle_deg_him < threshold).float().mean()  # 计算 Racc_5, Racc_15, Racc_30（旋转准确率）
                        predictions[f"Tacc_him_{threshold}"] = (
                                rel_tangle_deg_him < threshold).float().mean() 
                    Auc_30, normalized_histogram = calculate_auc(
                        rel_rangle_deg_him, rel_tangle_deg_him, max_threshold=30, return_list=True
                    )
                    auc_thresholds = [30, 10, 5, 3]
                    for auc_threshold in auc_thresholds:
                        predictions[f"Auc_{auc_threshold}"] = torch.cumsum(
                            normalized_histogram[:auc_threshold], dim=0
                        ).mean()
                    scene_name = batch["seq_name"][0]
                    AUC_scene_dict[scene_name] = torch.cumsum(normalized_histogram[:10], dim=0).mean()
                    predictions[f"Auc_scene_{scene_name}"] = AUC_scene_dict[scene_name]
                    is_demo = cfg.get("demo_json", False)
                    is_visual = cfg.visual_pose

                    if cfg.train.dataset == "AMD_test" and (is_demo or is_visual):
                        # --- 1. 公共准备工作 (路径与配置) ---
                        seq_name = sel_first_name[0]  # 提取序列名

                        # 内参矩阵 (假设固定)
                        K = np.array([[214.75555555, 0., 256.],
                                      [0., 286.34074074, 256.],
                                      [0., 0., 1.]])

                        # 数据集根目录
                        data_root = os.path.join(
                            os.path.dirname(__file__), "..", "datasets", "DCA_SpaceNet", "model1"
                        )
                        seq_dir = os.path.join(data_root, "testing", seq_name)

                        # 构建完整的图像路径列表 (公共逻辑)
                        full_image_paths = []
                        for frame_list in image_names:
                            if frame_list:
                                frame_name = frame_list[0]
                                full_path = os.path.join(seq_dir, os.path.join('000000/frame', frame_name))
                                full_image_paths.append(full_path)

                        num_images = len(full_image_paths)

                        # --- 2. 分支 A: 保存 Metrics JSON ---
                        if is_demo:
                            json_save_dir = os.path.join(seq_dir, "000000", "metrics")
                            os.makedirs(json_save_dir, exist_ok=True)
                            json_path = os.path.join(json_save_dir, "results.json")

                            # 辅助函数：转 Python 类型
                            def to_py(val):
                                if isinstance(val, torch.Tensor):
                                    return val.item() if val.numel() == 1 else val.tolist()
                                if isinstance(val, np.ndarray):
                                    return val.tolist()
                                return val

                            # 组装数据
                            traj_data = []
                            num_frames_pred = pred_cameras.R.shape[0]
                            for i in range(num_frames_pred):
                                traj_data.append({
                                    "frame_idx": i,
                                    "pred": {"R_quat": to_py(pred_cameras.R[i]), "T": to_py(pred_cameras.T[i])},
                                    "gt": {"R_quat": to_py(gt_cameras.R[i]), "T": to_py(gt_cameras.T[i])}
                                })

                            metrics_data = {
                                "R_avg": to_py(R_avg),
                                "T_avg": to_py(T_avg),
                                "Euler_Error_XYZ": [to_py(error_euler[2]), to_py(error_euler[1]),
                                                    to_py(error_euler[0])],
                                "Acc_5deg_XYZ": [to_py(acc_5_deg[2]), to_py(acc_5_deg[1]), to_py(acc_5_deg[0])],
                                "MSE_T_XYZ": [to_py(Tx_mse), to_py(Ty_mse), to_py(Tz_mse)],
                                "AUC": {f"Auc_{th}": to_py(predictions[f"Auc_{th}"]) for th in auc_thresholds},
                                "Accuracy_Him": {}
                            }
                            for th in thresholds:
                                metrics_data["Accuracy_Him"][f"Racc_{th}"] = to_py(predictions[f"Racc_him_{th}"])
                                metrics_data["Accuracy_Him"][f"Tacc_{th}"] = to_py(predictions[f"Tacc_him_{th}"])

                            final_json = {
                                "sequence_name": seq_name,
                                "metrics": metrics_data,
                                "trajectory": traj_data
                            }

                            try:
                                with open(json_path, 'w') as f:
                                    json.dump(final_json, f, indent=4)
                                print(f"✅ Metrics saved to: {json_path}")
                            except Exception as e:
                                print(f"❌ Failed to save metrics json: {e}")

                        # --- 3. 分支 B: 可视化保存图片 ---
                        if is_visual:
                            save_first_k_pose_images(
                                images_path=full_image_paths,
                                pred_cameras=pred_cameras,
                                gt_cameras=gt_cameras,
                                out_dir=os.path.join(seq_dir, "000000", "pose_vis"),
                                sel_name=seq_name,
                                K=K,
                                axis_len=5,
                                k=num_images,  # 动态使用上面计算好的数量
                                R_matrix_gt=R_matrix_gt.reshape(-1, 3, 3)
                            )

                            # 异常检测打印
                            if predictions["R_avg"] > 6:
                                print(f"⚠️ R_avg 超过阈值 ({predictions['R_avg']:.4f})")
                                print("seq_name =", seq_name)
                                print("image_names =", image_names)
        if training:
            stats.update(predictions, time_start=time_start, stat_set="train")
            if step % cfg.train.print_interval == 0:
                accelerator.print(stats.get_status_string(stat_set="train", max_it=max_it))
                accelerator.print(
                    f"  Batch Loss Trace:"
                    f"\n    loss_trans   : {predictions.get('loss_trans', 0):.6f}"
                    f"   loss_rot     : {predictions.get('loss_rot', 0):.6f}"
                    f"    loss_geo   : {predictions.get('loss_geo', 0):.6f}"
                )

        else:
            stats.update(predictions, time_start=time_start, stat_set="eval")
            if step % cfg.train.eval_print_interval == 0:
                accelerator.print(stats.get_status_string(stat_set="eval", max_it=max_it))

        if training:
            optimizer.zero_grad()  # 清空梯度
            if step % 100 == 0:  # 每100步检查一次
                with torch.autograd.detect_anomaly():
                    accelerator.backward(loss)
            else:
                accelerator.backward(loss)
            if cfg.train.clip_grad > 0:
                total_norm_before = accelerator.clip_grad_norm_(model.parameters(), cfg.train.clip_grad)
            optimizer.step()
            # memory_monitor.log(step, "After Optimizer Step")
            lr_scheduler.step()

            torch.cuda.empty_cache()

            # 清理中间变量
            del predictions
            del images
            del T_uvz
            del T_xyz
            del rotation
            del fl
            del pp
            del tracks
            del tracks_visibility
            if gt_cameras is not None:
                del gt_cameras
            if training:
                del loss

    print("x","y","z",x/len(dataloader),y/len(dataloader),z/len(dataloader))


    return True