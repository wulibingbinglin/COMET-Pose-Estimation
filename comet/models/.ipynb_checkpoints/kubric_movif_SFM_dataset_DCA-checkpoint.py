# 这个版本加上了 一开始的裁剪 最小框裁剪的步骤 r

import os
import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as Rt
from torchvision import transforms
import torchvision.transforms.functional as TF  # 推荐方式处理张量到PIL

from datasets.demo_loader import calculate_crop_parameters


def bbox_of_bright_regions(pil_img, thresh=30):
    """
    对 PIL.Image 做阈值分割（灰度化后），
    找到所有灰度 > thresh 的像素点，然后返回它们的 (xmin,ymin,xmax,ymax)。
    如果整图都不超过阈值，则返回整个图的 bbox。
    """
    # 转成 numpy 灰度图
    arr = np.array(pil_img.convert("L"))
    # 二值化
    mask = arr > thresh
    if not mask.any():
        # 全黑：直接返回整图
        w, h = pil_img.size
        return np.array([0, 0, w, h], dtype=int)
    ys, xs = np.nonzero(mask)
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    return np.array([xmin, ymin, xmax, ymax], dtype=int)


def make_bbox_square(old_bbox, size_to_fit):
    """
       在 old_bbox 的基础上，向四周各加 padding 使其长宽都扩展到 size_to_fit，并保持中心对齐。
       old_bbox: [xmin, ymin, xmax, ymax]
       size_to_fit: 目标正方形边长
    """
    new_bbox = np.array(old_bbox, dtype=np.float32)
    old_w = old_bbox[2] - old_bbox[0]
    old_h = old_bbox[3] - old_bbox[1]

    # 对高度进行 padding
    pad_h = (size_to_fit - old_h) / 2
    new_bbox[1] -= pad_h
    new_bbox[3] += pad_h

    # 对宽度进行 padding
    pad_w = (size_to_fit - old_w) / 2
    new_bbox[0] -= pad_w
    new_bbox[2] += pad_w

    # 最终转为整数坐标
    return new_bbox.astype(int)

def sample_uniform(total_frames, seq_len):
    # 任意间隔、无重复
    idx = np.random.choice(total_frames, size=seq_len, replace=False)
    return np.sort(idx).tolist()

def sample_with_max_gap(total_frames, seq_len):
    """
    在 total_frames 帧里选 seq_len 帧，
    保证相邻两帧的索引差（step）小于等于 max_gap：
      1) 随机选 step ∈ [1, max_gap]
      2) 随机选 start ∈ [0, total_frames - step*(seq_len-1) - 1]
      3) 返回 [start + i*step for i in 0..seq_len-1]
    """
    # 如果总帧太少，退回等间隔
    if total_frames < seq_len:
        return np.linspace(0, total_frames-1, seq_len, dtype=int).tolist()

    # 最大允许的步长
    max_step = (total_frames - 1) // (seq_len - 1)
    # print(max_step)
    # 在 [1, max_step] 里随机选一个
    #######################new#############################
    max_step = min(8, max_step)  # 硬上限8帧
    #######################new#############################

    if max_step < 1:  # 例如 total_frames 很小
        max_step = 1

    step = np.random.randint(1, max_step + 1)
    # print(max_step, step)
    # 计算合法的起点最大值：start + (seq_len-1)*step <= total_frames-1
    max_start = total_frames - (seq_len - 1) * step
    start = np.random.randint(0, max_start)

    # start = 2
    # step = 5
    # start = 20
    # step = 3
    # 生成等差序列
    idx = [start + i * step for i in range(seq_len)]
    return idx

class YTDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root,
        crop_size=(256, 256),
        seq_len=24,
        use_augs=False,
        split="train", # 如 "train" 表示训练集，"valid" 表示验证集）
    ):
        super().__init__()

        self.seq_len = seq_len
        self.crop_size = crop_size # 裁剪尺寸
        self.split = split #  |指定数据集的划分类型，例如训练集（"train"）或验证集（"valid"）
        if not os.path.exists(data_root):
            raise ValueError(f"Data root path does not exist: {data_root}")

        # 图像路径
        self.model_path = data_root

        if self.split == "valid":
            assert use_augs is False, "验证集不应使用增强"
        self.random_frame_rate = use_augs  # 或者通过 cfg 参数显式控制

        self.seq_names =  self.process_dataset_txt(self.model_path)
        # print(self.seq_names)

        print("found %d unique videos in %s" % (len(self.seq_names), self.model_path))

    def process_dataset_txt(self, model_path):
        all_sequences = []  # 存储所有序列的完整路径
        # model_path = os.path.join(model_path, "testing")
        model_path = os.path.join(model_path, "testing")
        # 3. 获取当前model下的所有sequence文件夹并排序
        sequences = [s for s in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, s)) and s.startswith("seq_")]
        
        exclude_seqs = {"seq_219", "seq_229", "seq_238", "seq_239"}
        sequences = [
            s for s in os.listdir(model_path)
            if os.path.isdir(os.path.join(model_path, s))
               and s.startswith("seq_")
               and int(s.split("_")[1]) < 233
               and s not in exclude_seqs
        ]
        exclude_seqs = {"seq_519", "seq_529", "seq_538", "seq_539"}
        sequences = [
                s for s in os.listdir(model_path)
                if os.path.isdir(os.path.join(model_path, s))
                   and s.startswith("seq_")
                   and int(s.split("_")[1]) < 533
                   and s not in exclude_seqs
            ]

        exclude_seqs = {"seq_819", "seq_829", "seq_838", "seq_839"}
        sequences = [
            s for s in os.listdir(model_path)
            if os.path.isdir(os.path.join(model_path, s))
               and s.startswith("seq_")
               and int(s.split("_")[1]) < 835
               and s not in exclude_seqs
        ]

        exclude_seqs = {"seq_1119", "seq_1129", "seq_1138", "seq_1139"}
        sequences = [
            s for s in os.listdir(model_path)
            if os.path.isdir(os.path.join(model_path, s))
               and s.startswith("seq_")
               and int(s.split("_")[1]) < 1135
               and s not in exclude_seqs
        ]

        sequences.sort(key=lambda x: int(x[4:]))
        # print(sequences)

        for seq in sequences:
            # 使用路径连接确保跨平台兼容性
            seq_path = os.path.join(model_path, seq)
            all_sequences.append(seq_path)

        return all_sequences

    def load_images_from_folder(self, seq_path):
        images_path = os.path.join(seq_path,"000000", "frame")
        gts_path = os.path.join(seq_path, "000000", "GroundTruth")
        masks_path = os.path.join(seq_path, "000000", "Mask")

        image_names = sorted(f for f in os.listdir(images_path) if f.startswith("frame_"))
        gt_names = sorted([f for f in os.listdir(gts_path) if f.startswith("obj_w2c_")])
        mask_names = sorted([f for f in os.listdir(masks_path) if f.startswith("mask_")])

        total_frames = len(image_names)
        if total_frames < self.seq_len:
            raise ValueError(f"Need {self.seq_len} frames, got {total_frames}")
        sel_inds = sample_with_max_gap(total_frames, self.seq_len)

        sel_names, images ,positions, orientations, list_bbox, rgbs, UVZ ,R_matrix= [], [], [], [], [], [], [],[]
        masks_raw = []  # 保存原始掩码（numpy uint8）
        #######对于选定的图片
        for ind in sel_inds:
            #############提供名字
            img_name = image_names[ind]

            #############提取image
            img = Image.open(os.path.join(images_path, img_name)).convert(
                "RGB") 

            # #############计算此帧对应的边界框
            mask_pil = Image.open(os.path.join(masks_path, mask_names[ind])).convert("L")  # 并将其转换为灰度图像（"L" 模式）。就只有01
            mask = np.array(mask_pil, dtype=np.uint8)
            masks_raw.append(mask)

            if np.any(mask > 0):  # 存在物体区域
                coords = cv2.findNonZero(mask.astype(np.uint8))
                x, y, w, h = cv2.boundingRect(coords)
                bbox = [x, y, x + w, y + h]  # 转换为xmin,ymin,xmax,ymax格式
            else:  # 全黑掩码
                h, w = mask.shape[:2]
                bbox = [0, 0, w, h]

            ###########提取t和r和uvz
            pose_matrix = np.loadtxt(os.path.join(gts_path, gt_names[ind]))
            if pose_matrix.shape != (4, 4):
                raise ValueError(f"{img_name} 不是有效的 4x4 矩阵")

            # S = np.diag([1, 1, -1])

            # 坐标系转换
            R_mat = pose_matrix[:3, :3] # 保证正交
            T_vec = pose_matrix[:3, 3]

            # 转四元数 (w,x,y,z)
            quat = Rt.from_matrix(R_mat).as_quat(scalar_first=True)

            # 相机内参
            fx, fy = 214.75555555, 286.34074074
            cx, cy = 256.0, 256.0

            # 相机投影到像素
            eps = 1e-6
            if abs(T_vec[2]) < eps:
                raise ZeroDivisionError(f"Tz≈0 in {gt_names[ind]}")
            u = (fx * T_vec[0] + cx * T_vec[2]) / T_vec[2]
            v = (fy * T_vec[1] + cy * T_vec[2]) / T_vec[2]

            #########存储各个变量###############
            R_matrix.append(R_mat)
            UVZ.append([u, v, T_vec[2]])
            sel_names.append(img_name)
            images.append(img)
            list_bbox.append(bbox)
            positions.append(T_vec.tolist())
            orientations.append(quat.tolist())

        ####################################### 1、（依赖所有掩码）用最小包围框crop 然后resize 记得上面也要改####################################
        ################计算所选中图片的最小外接矩阵############
        bbox_sequence = np.zeros(4)
        xmins, ymins, xmaxs, ymaxs = zip(*list_bbox)
        bbox_sequence[0] = min(xmins)
        bbox_sequence[1] = min(ymins)
        bbox_sequence[2] = max(xmaxs)
        bbox_sequence[3] = max(ymaxs)
        bbox_size = np.max([bbox_sequence[2] - bbox_sequence[0], bbox_sequence[3] - bbox_sequence[1]])
        max_size_with_margin = bbox_size * 1.3  # margin = 0.2 x max_dim
        margin = bbox_size * 0.15
        bbox_sequence = bbox_sequence + np.array([-margin, -margin, margin, margin])
        bbox_sequence_square = make_bbox_square(bbox_sequence, max_size_with_margin)

        # 裁剪 & 缩放到固定尺寸
        ratio = self.crop_size[0] / max_size_with_margin  # keep this value to predict translation later
        for image in images:
            cropped_img = image.crop(bbox_sequence_square)
            cropped_resized_img = cropped_img.resize(self.crop_size, Image.Resampling.LANCZOS)
            rgbs.append(cropped_resized_img)

        # ####################--- 仅对第一帧掩码做 crop + resize + 可选 dilate ---
        first_mask_np = masks_raw[0]  # uint8
        mask_pil_full = Image.fromarray(first_mask_np)
        # 将 NumPy 数组转换为 PIL 图像，便于使用 PIL 的 crop / resize 接口
        cropped_mask_pil = mask_pil_full.crop(bbox_sequence_square)  # 同 image.crop 的 bbox
        resized_first_mask_pil = cropped_mask_pil.resize(self.crop_size, Image.Resampling.NEAREST)
        first_mask_arr = np.array(resized_first_mask_pil, dtype=np.uint8)
        first_mask_bin = (first_mask_arr > 0).astype(np.uint8)

        # 转成 tensor 返回（float 0/1 与 bool 两种形式）
        # first_mask_tensor = torch.from_numpy(first_mask_bin).float()  # [H,W], float (0/1)
        first_mask_bool = torch.from_numpy(first_mask_bin.astype(bool))

        self.mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]

        video = torch.from_numpy(np.stack(rgbs, 0)).permute(0, 3, 1, 2).float()
        video = video / 255.0  # [0,1]
        video = (video - self.mean) / self.std  # 标准化到 ImageNet 分布
        T_pos = torch.from_numpy(np.array(positions)).float()
        R_matrix = torch.from_numpy(np.array(R_matrix)).float()
        R = torch.from_numpy(np.array(orientations)).float()
        T_pizza = torch.from_numpy(np.array(UVZ)).float()
        seq_name = os.path.basename(seq_path)

        return {"images": video, "T": T_pos, "R": R, "seq_name": seq_name,"T_uvz": T_pizza, "ratio": ratio,
                "image_names":sel_names, "first_mask":first_mask_bool, "R_matrix": R_matrix}
        # return {"images": video, "T": T_pos, "R": R, "seq_name": seq_name, "T_uvz": T_pizza, "ratio": ratio,
        #         "image_names": sel_names, "first_mask": first_mask_bool}

    def crop(self, rgbs, crop_size):
        H, W = rgbs[0].shape[:2]

        ############ spatial transform ############

        # 对验证集使用中心裁剪
        if self.split == "valid":
            y0 = (H - crop_size[0]) // 2
            x0 = (W - crop_size[1]) // 2
        else:
            # 训练集保持随机裁剪
            y0 = 0 if crop_size[0] >= H else (H- crop_size[0]) // 2
            x0 = 0 if crop_size[1] >= W else np.random.randint(0, W - crop_size[1])
        return rgbs[:, y0:y0 + self.crop_size[0],
                   x0:x0 + self.crop_size[1]] # 对列表中每一帧 rgb，按照行区间 [y0, y0+crop_h) 和列区间 [x0, x0+crop_w) 进行切片

    def __len__(self):
        return len(self.seq_names)

    def __getitem__(self, index):
        seq_name = self.seq_names[index] # 根据给定的 index，获取对应的序列名称（seq_name）
        sample= self.load_images_from_folder(seq_name)

        return sample

def test_spark_dataset():
    # 1. 设置数据根目录（请根据实际路径修改）
    data_root = "../datasets/AMD"

    # 2. 实例化数据集
    dataset = YTDataset(
        data_root=os.path.join(data_root, "AMD_train"),
        crop_size=[256, 256],
        seq_len=10,
        use_augs=True,
        split="train",
    )

    print(f"Dataset length: {len(dataset)} sequences")

    # 3. 使用 DataLoader（可选），这里我们直接索引第一个样本
    sample = dataset[0]

    images = sample["images"]      # Tensor, shape: [T, C, H, W]
    positions = sample["T"]        # Tensor, shape: [T, 3]
    orientations = sample["R"]     # Tensor, shape: [T, 4]
    seq_name = sample["seq_name"]  # 对应的序列名称，如 "s001"

    # 4. 打印各项信息，验证正确加载
    print(f"Sequence name: {seq_name}")
    print(f"  images   tensor shape: {images.shape}")
    print(f"  positions tensor shape: {positions.shape}")
    print(f"  orientations tensor shape: {orientations.shape}")

    print(positions)
    print(f"  orientations: {orientations}")
    # # 5. 若需遍历前 N 帧的第一个像素值（示例）
    # for t in range(min(3, images.shape[0])):  # 前 3 帧
    #     pixel = positions[t, :, 0, 0]  # 通道 C 的第一个像素
    #     print(f"  frame {t} first pixel (C): {pixel.tolist()}")

if __name__ == "__main__":
    test_spark_dataset()