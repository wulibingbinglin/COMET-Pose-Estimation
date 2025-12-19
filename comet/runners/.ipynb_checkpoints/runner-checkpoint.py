# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os  # 提供操作系统相关功能，例如文件路径管理、文件操作等。
import sys  # 用于与 Python 解释器交互，比如获取命令行参数或修改模块路径。
import copy  # 提供浅拷贝和深拷贝功能，避免直接修改原始数据对象。

import torch  # PyTorch 是深度学习框架，这里用于张量计算、深度学习模型调用以及 GPU 加速。
import pycolmap  # Python 封装的 COLMAP 工具，用于 SfM（Structure-from-Motion）流程中的相机参数估计和三维点云生成。
import datetime  # 提供日期和时间处理功能，比如记录运行时间或日志。
import time  # 提供时间测量功能，例如记录程序执行的时间。

import numpy as np  # 高效的数值计算库，用于处理矩阵和数组操作，是深度学习和 SfM 的基础工具。
from visdom import Visdom  # 用于实时可视化工具，动态展示重建结果或模型训练中的关键信息。
from torch.cuda.amp import autocast  # 自动混合精度模块，用于加速 GPU 计算，降低显存占用。

from hydra.utils import instantiate  # 从 Hydra 配置文件中动态初始化类和对象。

# 引入特征提取算法，SfM 的核心部分，用于图像间的关键点匹配。
from lightglue import SuperPoint, SIFT, ALIKED
# SuperPoint: 基于深度学习的特征提取器。
# SIFT: 经典的特征检测算法。
# ALIKED: 高效的轻量化特征提取方法。

from collections import defaultdict  # 带有默认值的字典，用于存储匹配信息，避免未初始化键访问时报错。

# 可视化工具，用于展示 SfM 的重建过程，包括相机轨迹、特征点匹配等。
from vggsfm.utils.visualizer import Visualizer

# 初步估计相机参数的核心方法，例如内参和外参的计算，SfM 的重要步骤。
from vggsfm.two_view_geo.estimate_preliminary import (
    estimate_preliminary_cameras,
)

# 引入多个实用工具函数，用于辅助 SfM 过程的细节操作。
from vggsfm.utils.utils import (
    write_array,  # 将数据保存为数组文件，便于后续处理。
    generate_grid_samples,  # 在图像中生成规则网格点，用于匹配验证。
    generate_rank_by_midpoint,  # 通过中点法对特征点进行排序。
    generate_rank_by_dino,  # 基于视觉 Transformer (DINO) 的特征点排序。
    generate_rank_by_interval,  # 根据采样间隔排序特征点。
    calculate_index_mappings,  # 计算特征点索引的映射关系。
    extract_dense_depth_maps,  # 从图像中提取密集的深度图。
    align_dense_depth_maps,  # 对齐深度图，确保多张图像的深度一致性。
    switch_tensor_order,  # 改变张量的维度顺序，便于不同框架处理。
    sample_subrange,  # 从大范围数据中采样子集。
    average_camera_prediction,  # 对多帧相机预测结果进行平均，提升稳定性。
    create_video_with_reprojections,  # 生成包含重投影的可视化视频。
    save_video_with_reprojections,  # 保存包含重投影效果的视频。
)

# 三角测量方法：从多视图匹配点中生成三维点云，SfM 的核心几何操作。
from vggsfm.utils.triangulation import triangulate_tracks
# 提供相机模型转换和三维点过滤的辅助工具。
from vggsfm.utils.triangulation_helpers import cam_from_img, filter_all_points3D

# 选择性引入 Poselib（高性能相机位姿估计库）。
try:
    import poselib
    from vggsfm.two_view_geo.estimate_preliminary import (
        estimate_preliminary_cameras_poselib,  # 使用 Poselib 加速相机参数估计。
    )
    print("Poselib is available")
except:
    print("Poselib is not installed. Please disable use_poselib")

# 选择性引入 PyTorch3D（用于三维点云可视化和渲染）。
try:
    from pytorch3d.structures import Pointclouds  # 三维点云数据结构。
    from pytorch3d.vis.plotly_vis import plot_scene  # 用于绘制三维场景。
    from pytorch3d.renderer.cameras import (
        PerspectiveCameras as PerspectiveCamerasVisual,  # 透视相机模型，可视化用途。
    )
except:
    print("PyTorch3d is not available. Please disable visdom.")


class VGGSfMRunner:
    def __init__(self, cfg):
        """
        A runner class for the VGGSfM (Structure from Motion) pipeline.

        This class encapsulates the entire SfM process, including model initialization,
        sparse and dense reconstruction, and visualization.

        Args:
            cfg: Configuration object containing pipeline settings.
        """

        self.cfg = cfg

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.build_vggsfm_model()
        self.camera_predictor = self.vggsfm_model.camera_predictor
        self.track_predictor = self.vggsfm_model.track_predictor
        self.triangulator = self.vggsfm_model.triangulator

        if cfg.dense_depth:
            self.build_monocular_depth_model()
        # 构建单目深度估计模型。也就是说，当程序决定启用稠密深度估计时，它就会调用这个方法来生成一个深度估计模型。单目指的是“单眼”，
        # 也就是只用一张图片或一个视角来进行深度(每个像素到摄像头的距离)估计，而不是多张图片或者立体视觉。

        if cfg.viz_visualize:
            self.build_visdom()

        # Set up mixed precision
        assert cfg.mixed_precision in ("None", "bf16", "fp16")
        # cfg.mixed_precision：表示从配置对象 cfg 中获取混合精度的设置。这通常是 "None"（不使用混合精度）、
        # "bf16"（bfloat16 精度）或 "fp16"（float16 精度）中的一种。
        self.dtype = {
            "None": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }.get(cfg.mixed_precision, None)
        """作用：根据配置 cfg.mixed_precision 的值，选择相应的数据类型（dtype）。这里使用了 Python 的字典 get() 方法来映射混合精度设置到 PyTorch 的数据类型。
        如果 cfg.mixed_precision 为 "None"，则选择 torch.float32（32位浮动数，是常规的精度）。
        如果 cfg.mixed_precision 为 "bf16"，则选择 torch.bfloat16（bfloat16 精度）。
        如果 cfg.mixed_precision 为 "fp16"，则选择 torch.float16（float16 精度）。
        
        get() 方法：这是字典对象的方法，用于返回字典中指定键（cfg.mixed_precision）对应的值。如果键不存在，则返回第二个参数（在这里是 None）。
        self.dtype：将选择的精度类型存储在实例的 dtype 属性中，以供后续使用。"""

        if self.dtype is None:
            raise NotImplementedError(
                f"dtype {cfg.mixed_precision} is not supported now"
            )

        # Remove the pixels too close to the border
        self.remove_borders = 4 # 在计算机视觉中，图像边缘的像素常常会因为各种原因（如相机畸变、视角变化等）导致不稳定或者不准确的计算。
                                # 去除这些边缘像素可以提高后续处理的精度，尤其是在特征匹配、深度估计等任务中。

    def build_vggsfm_model(self):
        """
        构建 VGGSfM 模型并加载checkpoint，
        Checkpoint是用于描述在每次训练后保存模型参数（权重）的惯例或术语
        Builds the VGGSfM model and loads the checkpoint.

        Initializes the VGGSfM model and loads the weights from a checkpoint.
        The model is then moved to the appropriate device and set to evaluation mode.
        """

        print("Building VGGSfM")

        vggsfm = instantiate(self.cfg.MODEL, _recursive_=False, cfg=self.cfg)
        # 这一行代码使用 instantiate 函数根据配置对象（cfg）中的信息动态实例化 VGGSfM 模型
        """
        self.cfg.MODEL 内容：

        它是一个包含模型配置的字典（或类似的配置对象），定义了目标类路径 _target_ 以及初始化该类所需的参数。
        cfg=self.cfg 的作用：
        
        它是通过 **kwargs 传入的额外参数，用于在调用目标类的构造函数时，动态传递配置。
        这些参数可以在目标类的 __init__ 方法中被显式接收。
        覆盖关系：
        
        如果 self.cfg.MODEL 中已经包含了一个键 cfg，而此处又通过 cfg=self.cfg 传入了同名键，\
        则 cfg=self.cfg 会覆盖配置中原有的 cfg 值。
        """

        """
        hydra.utils.instantiate(config, *args, **kwargs)
        config:        
        配置对象，可以是一个字典或通过 Hydra 定义的配置结构。
        必须包含键 _target_，其值是目标类或函数的完整路径（模块名 + 类名或函数名）。
        其他键将被视为目标类或函数的初始化参数。
        
        *args 和 **kwargs:        
        用于在调用目标类或函数时传递额外的非配置参数。
        它们会覆盖 config 中的同名参数。
        
        _recursive_ (特殊参数，配置中常用):        
        控制是否递归解析嵌套配置。
        当 _recursive_ = False 时，instantiate() 不会递归实例化子模块配置，而是将子模块配置作为普通字典传入目标对象。"""

        if self.cfg.auto_download_ckpt: # 如果 auto_download_ckpt 为 True，则会通过调用 from_pretrained 方法从预训练模型中加载训练好的权重。
            vggsfm.from_pretrained(self.cfg.model_name)
        else:
            checkpoint = torch.load(self.cfg.resume_ckpt) # 这行代码从本地磁盘加载 PyTorch 检查点文件。
            # torch.load() 是 PyTorch 中的函数，用于加载保存的模型或张量。通常，在训练过程中，
            # 我们会定期保存模型的状态（权重、优化器状态等），以便以后恢复训练或进行推理。
            vggsfm.load_state_dict(checkpoint, strict=True)
            # checkpoint：是从 torch.load() 加载的模型权重，通常是一个字典，包含了模型层的权重和偏置。
            # strict=True 表示模型必须严格匹配加载的权重，如果检查点中的某些层在模型中不存在或维度不匹配，将引发错误
        self.vggsfm_model = vggsfm.to(self.device).eval()
        # eval()：将模型设置为评估模式，这意味着在推理时
        # 模型会关闭像 Dropout 和 BatchNorm 等训练过程中使用的特性。这样可以确保推理的一致性
        # self.vggsfm_model 变量记录了最终的、初始化并准备好进行推理的模型对象。
        print("VGGSfM built successfully")

    def build_monocular_depth_model(self):
        """
        Builds the monocular depth model and loads the checkpoint.

        This function initializes the DepthAnythingV2 model,
        downloads the pre-trained weights from a URL, and loads these weights into the model.
        The model is then moved to the appropriate device and set to evaluation mode.
        """
        # Import DepthAnythingV2 inside the function to avoid unnecessary imports

        parent_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        # os.path.dirname(__file__) 返回当前脚本文件所在的目录路径。
        # os.path.join(path, *paths)
        #
        # 拼接路径字符串，生成新的路径。
        # os.path.join(os.path.dirname(__file__), "..", "..") 等价于将当前文件目录向上两级
        # os.path.abspath(path) 将相对路径转换为绝对路径。
        sys.path.append(parent_path) # 将新的路径添加到 sys.path，这是 Python 搜索模块时的路径列表。
        from dependency.depth_any_v2.depth_anything_v2.dpt import (
            DepthAnythingV2,
        ) # 因为当前目录前两级不在，所以报错

        print("Building DepthAnythingV2")
        model_config = {
            "encoder": "vitl",
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
        }
        depth_model = DepthAnythingV2(**model_config)
        # DepthAnythingV2(**kwargs)
        # ** 是 Python 的关键字参数解包操作符，表示将字典 model_config 中的键值对作为参数传入构造函数。
        # 等价于 DepthAnythingV2(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024])。 下载并加载预训练权重
        _DEPTH_ANYTHING_V2_URL = "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth"
        checkpoint = torch.hub.load_state_dict_from_url(_DEPTH_ANYTHING_V2_URL)
        depth_model.load_state_dict(checkpoint)
        self.depth_model = depth_model.to(self.device).eval() # 将模型移动到设备并设置评估模式
        print(f"DepthAnythingV2 built successfully")

    def build_visdom(self):
        """
        Set up a Visdom server for visualization.
        """
        self.viz = Visdom()

    def run(
        self,
        images,
        masks=None,# none
        original_images=None,
        image_paths=None,
        crop_params=None,
        query_frame_num=None,# none
        seq_name=None,
        output_dir=None,
    ):
        """
        Executes the full VGGSfM pipeline on a set of input images.

        This method orchestrates the entire reconstruction process, including sparse
        reconstruction, dense reconstruction (if enabled), and visualization.

        Args:
            images (torch.Tensor): Input images with shape Tx3xHxW or BxTx3xHxW, where T is
                the number of frames, B is the batch size, H is the height, and W is the
                width. The values should be in the range (0,1).
            masks (torch.Tensor, optional): Input masks with shape Tx1xHxW or BxTx1xHxW.
                Binary masks where 1 indicates the pixel is filtered out.
                过滤噪声
            original_images (dict, optional): Dictionary with image basename as keys and original
                numpy images (rgb) as values.
            image_paths (list of str, optional): List of paths to input images. If not
                provided, you can use placeholder names such as image0000.png, image0001.png.
            crop_params (torch.Tensor, optional): A tensor with shape Tx8 or BxTx8. Crop parameters
                indicating the mapping from the original image to the processed one (We pad
                and resize the original images to a fixed size.).
                裁剪、归一化、缩放、对齐
            query_frame_num (int, optional): Number of query frames to be used. If not
                specified, will use self.cfg.query_frame_num.
            seq_name (str, optional): Name of the sequence.
            output_dir (str, optional): Directory to save the output. If not specified,
                the directory will be named as f"{seq_name}_{timestamp}".
        Returns:
            dict: A dictionary containing the predictions from the reconstruction process.
        """
        if output_dir is None:
            now = datetime.datetime.now()
            timestamp = now.strftime("%Y%m%d_%H%M")
            # 将当前时间格式化为YYYYMMDD_HHMM格式的字符串。
            output_dir = f"{seq_name}_{timestamp}" # 使用 strftime 方法将当前的日期时间格式化成指定的字符串格式

        with torch.no_grad():
            images = move_to_device(images, self.device)
            masks = move_to_device(masks, self.device)
            crop_params = move_to_device(crop_params, self.device)
            # 将输入图像（images）、掩码（masks）和裁剪参数（crop_params）移存到指定的设备

            # Add batch dimension if necessary 这里的“帧”是指图像序列中的每一张图像。以视频为例，一个视频包含多个连续的帧，
            # 通常这些帧在时间上有一定的顺序。在推理或训练过程中，你可以将这些连续的图像帧当作一个图像序列（T 帧），每个图像序列都是一个批次的一部分。
            # 分批次处理并不会改变三维重建的精度，原因是这些批次是在推理过程中独立处理的，最后的点云或相机姿态会通过合并各批次的结果得到最终输出。
            # 每个批次内的数据被独立地处理，最终的重建结果不会受到分批次输入的影响，因为每个图像在模型中单独进行推理，处理顺序不会影响结果。
            if len(images.shape) == 4:
                images = add_batch_dimension(images) # 之前仅仅是四个维度，是每个batch，现在要去添加批次维度(1,10,3,1024,1024)
                masks = add_batch_dimension(masks) # none
                crop_params = add_batch_dimension(crop_params) # (1,10,8)

            if query_frame_num is None:
                query_frame_num = self.cfg.query_frame_num # 让配置信息直接给予查询帧数量

            # Perform sparse reconstruction
            predictions = self.sparse_reconstruct(
                images,
                masks=masks,
                crop_params=crop_params,
                image_paths=image_paths,
                query_frame_num=query_frame_num,
                seq_name=seq_name,
                output_dir=output_dir,
            )

            # Save the sparse reconstruction results
            if self.cfg.save_to_disk:
                self.save_sparse_reconstruction(
                    predictions, seq_name, output_dir
                )
                
                if predictions["additional_points_dict"] is not None:
                    additional_dir = os.path.join(output_dir, "additional")
                    # additional_dir = os.path.join(output_dir, "additional")：
                    # 在 output_dir 目录下创建一个名为 additional 的子目录，用于存储额外的数据。
                    os.makedirs(additional_dir, exist_ok=True)
                    # 确保 additional_dir 目录存在，如果不存在则创建。
                    torch.save(predictions["additional_points_dict"], os.path.join(additional_dir, "additional_points_dict.pt"))
                    # 将额外的点信息保存到 additional_points_dict.pt 文件中。


            # Extract sparse depth and point information if needed for further processing
            if self.cfg.dense_depth or self.cfg.make_reproj_video:
                predictions = (
                    self.extract_sparse_depth_and_point_from_reconstruction(
                        predictions)
                )
                # 调用此函数从稀疏重建结果中提取深度信息和点云信息，并将其返回。predictions
                # 作为输入，包含了重建的所有信息。

            # Perform dense reconstruction if enabled
            if self.cfg.dense_depth:
                predictions = self.dense_reconstruct(
                    predictions, image_paths, original_images
                )

                # Save the dense depth maps
                if self.cfg.save_to_disk:
                    self.save_dense_depth_maps(
                        predictions["depth_dict"], output_dir
                    )

            # Create reprojection video if enabled
            if self.cfg.make_reproj_video:
                max_hw = crop_params[0, :, :2].max(dim=0)[0].long()
                """整个表达式 max_hw = crop_params[0, :, :2].max(dim=0)[0].long() 实现的过程如下：

                从 crop_params 中取出第一个 batch，并只选择每行的前两列，得到一个 2x2 的子 Tensor。
                沿着行方向（dim=0）对每列求最大值，得到每列的最大值。
                选择最大值的部分（max(dim=0)[0]），并将结果转换为 torch.int64 类型。"""
                video_size = (max_hw[0].item(), max_hw[1].item())
                # ed.
                # max_hw = torch.tensor([300, 400])
                # max_hw[0]  # 输出：tensor(300)
                # item() 是 PyTorch 中的一个方法，能够将一个包含单一值的张量转换为 Python 的数字类型
                img_with_circles_list = self.make_reprojection_video(
                    predictions, video_size, image_paths, original_images
                )
                predictions["reproj_video"] = img_with_circles_list
                if self.cfg.save_to_disk:
                    self.save_reprojection_video(
                        img_with_circles_list, video_size, output_dir
                    )

            # Visualize the 3D reconstruction if enabled
            if self.cfg.viz_visualize:
                self.visualize_3D_in_visdom(predictions, seq_name, output_dir)

            if self.cfg.gr_visualize:
                self.visualize_3D_in_gradio(predictions, seq_name, output_dir)

            return predictions

    def sparse_reconstruct(
        self,
        images,
        masks=None,
        crop_params=None,
        query_frame_num=3,
        image_paths=None,
        seq_name=None,
        output_dir=None,
        dtype=None, # 指定计算时的数据类型
        back_to_original_resolution=True, # 是否返回到原始分辨率，默认值为 True。
    ):
        """
        Perform sparse reconstruction on the given images.

        This function implements the core SfM pipeline, including:
        1. Selecting query frames
        2. Estimating camera poses
        3. Predicting feature tracks across frames
        4. Triangulating 3D points
        5. Performing bundle adjustment

        Args:
            images (torch.Tensor): A tensor of shape (B, T, C, H, W) representing a batch of images. 可定位到某个像素
            masks (torch.Tensor): A tensor of shape (B, T, 1, H, W) representing masks for the images.
            crop_params (torch.Tensor): A tensor of shape (B, T, 8), indicating the mapping from the original image
                                    to the processed one (We pad and resize the original images to a fixed size.).
                                    top: 裁剪区域的上边界（相对于原始图像的顶部的距离）。
                                    left: 裁剪区域的左边界（相对于原始图像的左边的距离）。
                                    height: 裁剪区域的高度。
                                    width: 裁剪区域的宽度。
            query_frame_num (int): The number of query frames to use for reconstruction. Default is 3.
            image_paths (list): A list of image file paths corresponding to the input images.
            seq_name (str): The name of the sequence being processed.
            output_dir (str): The directory to save the output files.
            dtype (torch.dtype): The data type to use for computations.

            NOTE During inference we force B=1 now.
        Returns:
            dict: A dictionary containing the reconstruction results, including camera parameters and 3D points.
        """

        print(f"Run Sparse Reconstruction for Scene {seq_name}")
        batch_num, frame_num, image_dim, height, width = images.shape
        device = images.device
        reshaped_image = images.reshape(
            batch_num * frame_num, image_dim, height, width
        )
        visual_dir = os.path.join(output_dir, "visuals") # os.path.join()：拼接路径，自动适配操作系统路径分隔符。

        if dtype is None: # 如果未提供数据类型 dtype，使用类属性 self.dtype 作为默认值。
            dtype = self.dtype

        predictions = {}

        # ################Find the query frames using DINO or frame names###########
        with autocast(dtype=dtype):
            if self.cfg.query_by_midpoint: # 使用“中点”策略来选择查询帧
                query_frame_indexes = generate_rank_by_midpoint(frame_num)
            elif self.cfg.query_by_interval: # 按“间隔”策略选择查询帧
                query_frame_indexes = generate_rank_by_interval(
                    frame_num, (frame_num // query_frame_num + 1)
                )
            else:
                query_frame_indexes = generate_rank_by_dino( # DINO
                    reshaped_image, self.camera_predictor, frame_num
                ) # [5, 3, 4, 1, 6, 0, 2, 7]
        # 功能：根据不同策略选择查询帧的索引：
        # query_by_midpoint：基于帧的中点生成索引。该策略适用于那些需要从视频或序列中获取代表性帧的场景，这些帧通常能代表整个序列的中间状态
        # query_by_interval：基于固定间隔生成索引。适用于那些希望均匀分布查询帧的场景。通过这种策略，可以确保查询帧分布在整个序列或视频的不同位置，避免过度集中在某一部分。
        # 默认情况：使用DINO模型与相机预测器生成查询帧索引。DINO（Distillation with No Labels）是一种视觉模型，能基于图像的视觉特征（比如物体、背景等）来选择具有代表性的帧。

        # Extract base names from image paths
        image_paths = [os.path.basename(imgpath) for imgpath in image_paths]
        # 输入：image_paths = ["dir/image1.png", "dir/image2.png"]
        # 输出：["image1.png", "image2.png"] 叫做照片名字了都

        center_order = None # center_order 用于存储新的帧顺序
        # Reorder frames if center_order is enabled
        if self.cfg.center_order: # 改变第一个帧为查询帧,若不改变 就不管
            # The code below switchs the first frame (frame 0) to the most common frame
            center_frame_index = query_frame_indexes[0] # 第一个查询帧的索引
            if center_frame_index != 0: # 说明查询帧在现在图片序列所在的位置上不是第一个
                center_order = calculate_index_mappings( # 查询帧与第一帧位置互换，原先是（1，2，3，4），查询真是3，改完之后的order为（3，2，1.4）
                    center_frame_index, frame_num, device=device
                )

                images, crop_params, masks = switch_tensor_order(
                    [images, crop_params, masks], center_order, dim=1
                )  # 根据 center_order 重新排列 images、crop_params 和 masks 张量的顺序。
                reshaped_image = switch_tensor_order( #(BN)CHW
                    [reshaped_image], center_order, dim=0
                )[0] # 重新排列 reshaped_image 张量的顺序，并取出第一个元素。 以后都按这个顺序走了

                image_paths = [
                    image_paths[i] for i in center_order.cpu().numpy().tolist()
                ]
                # 功能：根据新的顺序重新排列 image_paths。
                # 语法解析：
                # center_order.cpu()：将 center_order 张量移到 CPU 上。
                # .numpy()：转换为 NumPy 数组。
                # .tolist()：将 NumPy 数组转换为 Python 列表。
                # 列表推导式：按新顺序遍历 image_paths 并重新排列。

                # Also update query_frame_indexes:
                query_frame_indexes = [
                    center_frame_index if x == 0 else x
                    for x in query_frame_indexes
                ]
                # 此为调换查询帧为0的位置和为center_frame_index的位置，因为现在想让center_frame_index在第0位

                # 将查询帧索引中的第0帧替换为中心帧索引
                query_frame_indexes[0] = 0 # 现在第一个查询帧就是 所有 信息的第一个帧

        # Select only the specified number of query frames
        query_frame_indexes = query_frame_indexes[:query_frame_num]

        # Predict Camera Parameters by camera_predictor
        if self.cfg.avg_pose:
            # Conduct several times with different frames as the query frame
            # self.camera_predictor is super fast and this is almost a free-lunch
            # 系统会使用多个查询帧（在 query_frame_indexes 中指定）对相机参数进行预测，
            # 然后对多个预测结果取平均。这种方法可以减少单一帧的误差，提高预测的鲁棒性。
            pred_cameras = average_camera_prediction(
                self.camera_predictor,
                reshaped_image,
                batch_num, # 1
                query_indices=query_frame_indexes,
            ) # 如果启用 avg_pose，将使用多个不同帧作为查询帧进行相机预测，并取平均结果。
        else:
            pred_cameras = self.camera_predictor(
                reshaped_image, batch_size=batch_num
            )["pred_cameras"]  # 调用相机预测模型，预测过程将基于单个查询帧进行。。

        # Prepare image feature maps for tracker 用的是cotracker 1 8 C=128 h=128 w=128
        fmaps_for_tracker = self.track_predictor.process_images_to_fmaps(images)

        # Calculate bounding boxes if crop parameters are provided
        if crop_params is not None: # width, height, crop width, scale, and adjusted bounding box coordinates  [x_min, y_min, x_max, y_max].
            bound_bboxes = crop_params[:, :, -4:-2].abs().to(device) # 1 8 2 q 取调整之后图片的xmin，ymin 取绝对值
            """ 切片语法规则总结 
            1\冒号 :：
            代表取该维度的所有元素。
            2\索引范围 [start:stop]：
            start：开始索引（包含该索引）。
            stop：结束索引（不包含该索引）。
            如果 start 或 stop 是负数，则从末尾开始计算（-1 表示倒数第一个元素，-2 表示倒数第二个元素，以此类推）。"""
            # also remove those near the boundary
            bound_bboxes[bound_bboxes != 0] += self.remove_borders # 1 8 2 bound_bboxes != 0 只对 非零 边界值进行调整+4，避免不必要的操作
            bound_bboxes = torch.cat(
                [bound_bboxes, reshaped_image.shape[-1] - bound_bboxes], dim=-1
            )# 计算得出在增加边框之后的 右下角的坐标 此时和crop_params中的数或许差一个边框的距离 1 8 4 并连接在一起  得到加入边框的[x_min, y_min, x_max, y_max].

        # Predict tracks
        with autocast(dtype=dtype): # autocast(dtype=dtype)：启用 混合精度计算，加速推理并减少显存占用。
            pred_track, pred_vis, pred_score = predict_tracks(
                self.cfg.query_method, # "aliked"
                self.cfg.max_query_pts, # max_query_pts: 2048
                self.track_predictor,
                images,
                masks,
                fmaps_for_tracker, # cotracker 1 8 128 128 128
                query_frame_indexes,# 5 3 4
                self.cfg.fine_tracking, # True
                bound_bboxes, # 四个边界点坐标
            )

            # Complement non-visible frames if enabled
            if self.cfg.comple_nonvis: # 调用 comple_nonvis_frames 函数对不可见帧的信息进行补充。
                pred_track, pred_vis, pred_score = comple_nonvis_frames(
                    self.cfg.query_method,
                    self.cfg.max_query_pts, # 2048
                    self.track_predictor,
                    images,
                    masks,
                    fmaps_for_tracker, # 1 8 2048 2048
                    [pred_track, pred_vis, pred_score],
                    self.cfg.fine_tracking, #true
                    bound_bboxes,
                )


        # Visualize tracks as a video if enabled
        if self.cfg.visual_tracks:
            vis = Visualizer(save_dir=visual_dir, linewidth=1) # 实例化一个 Visualizer 类，并指定保存目录 visual_dir 和线宽参数 linewidth=1。
            vis.visualize( # 将输入图像乘以 255，将原本归一化到 [0,1] 的像素值恢复到 [0,255] 范围，便于显示
                images * 255, pred_track, pred_vis[..., None], filename="track"
            ) # 调用可视化方法，将输入图像、预测轨迹、可见性等数据渲染成视频。

        torch.cuda.empty_cache() # 功能：清空 GPU 上未使用的缓存内存。

        # Force predictions in padding areas as non-visible
        if crop_params is not None:
            hvis = torch.logical_and( # pred_track[..., 1]：预测轨迹中的 y 坐标后两个分别是边界框在 y 方向上的最小值和最大值。通过 torch.logical_and 判断预测点的 y 坐标是否处于该范围内，结果存储在 hvis 中
                pred_track[..., 1] >= bound_bboxes[:, :, 1:2],
                pred_track[..., 1] <= bound_bboxes[:, :, 3:4],
            )
            wvis = torch.logical_and( # x
                pred_track[..., 0] >= bound_bboxes[:, :, 0:1],
                pred_track[..., 0] <= bound_bboxes[:, :, 2:3],
            )
            force_vis = torch.logical_and(hvis, wvis)
            pred_vis = pred_vis * force_vis.float() # # 只有同时满足 x 和 y 两个方向条件的预测点才被视为有效的（可见的）

        if self.cfg.use_poselib: # 根据配置 self.cfg.use_poselib 来决定使用哪个函数来估计初步的相机参数。
            estimate_preliminary_cameras_fn = (
                estimate_preliminary_cameras_poselib
            )
        else:
            estimate_preliminary_cameras_fn = estimate_preliminary_cameras

        # Estimate preliminary_cameras by recovering fundamental/essential/homography matrix from 2D matches
        # By default, we use fundamental matrix estimation with 7p/8p+LORANSAC
        # All the operations are batched and differentiable (if necessary) 为后续的三维重建或相机姿态优化提供基础数据
        # except when you enable use_poselib to save GPU memory
        _, preliminary_dict = estimate_preliminary_cameras_fn(
            pred_track, # 预测的轨迹 目标是从预测的2D匹配中恢复出基础矩阵（fundamental matrix）、本质矩阵（essential matrix）或单应性矩阵（homography matrix）
            pred_vis,
            width, # 1024
            height,  # 1024
            tracks_score=pred_score,
            max_error=self.cfg.fmat_thres, # 4
            loopresidual=True,
        )

        # Perform triangulation and bundle adjustment
        with autocast(dtype=torch.float32): # ：启用自动混合精度计算（AMP）来提高计算效率。
            (
                extrinsics_opencv,
                intrinsics_opencv,
                extra_params,
                points3D,
                points3D_rgb,
                reconstruction,
                valid_frame_mask,
                valid_2D_mask,
                valid_tracks,
            ) = self.triangulator(
                pred_cameras,
                pred_track,
                pred_vis,
                images,
                preliminary_dict,
                pred_score=pred_score,
                BA_iters=self.cfg.BA_iters,
                shared_camera=self.cfg.shared_camera,
                max_reproj_error=self.cfg.max_reproj_error,
                init_max_reproj_error=self.cfg.init_max_reproj_error,
                extract_color=self.cfg.extract_color,
                robust_refine=self.cfg.robust_refine,
                camera_type=self.cfg.camera_type,
            )

        
        additional_points_dict = None # 变量将在后续步骤中用来存储额外三维点的相关信息。
        
        if self.cfg.extra_pt_pixel_interval > 0: # 表示需要三角化额外的点
            additional_points_dict = self.triangulate_extra_points(
                images,
                masks,
                fmaps_for_tracker,
                bound_bboxes,
                intrinsics_opencv,
                extra_params,
                extrinsics_opencv,
                image_paths,
                frame_num,
            ) # 三角化额外的 3D 点，并将结果存储在 additional_points_dict 中。
            additional_points3D = torch.cat(
                [
                    additional_points_dict[img_name]["points3D"]
                    for img_name in image_paths
                ],
                dim=0,
            )
            additional_points3D_rgb = torch.cat(
                [
                    additional_points_dict[img_name]["points3D_rgb"]
                    for img_name in image_paths
                ],
                dim=0,
            )
            # 拼接每张图像中的 3D 点和 RGB 点，形成完整的额外 3D 点数据。

            additional_points_dict["sfm_points_num"] = len(points3D)
            additional_points_dict["additional_points_num"] = len(additional_points3D)

            if self.cfg.concat_extra_points: # 将额外的 3D 点和 RGB 点连接到现有的点云中
                additional_points3D_numpy = additional_points3D.cpu().numpy()
                additional_points3D_rgb_numpy = (
                    (additional_points3D_rgb * 255).long().cpu().numpy()
                )
                for extra_point_idx in range(len(additional_points3D)):
                    reconstruction.add_point3D(
                        additional_points3D_numpy[extra_point_idx],
                        pycolmap.Track(),
                        additional_points3D_rgb_numpy[extra_point_idx],
                    )
                    
                points3D = torch.cat([points3D, additional_points3D], dim=0)
                points3D_rgb = torch.cat(
                    [points3D_rgb, additional_points3D_rgb], dim=0
                )
                # 将原始的 3D 点和额外的 3D 点拼接起来，形成最终的 3D 点云数据。

        if self.cfg.filter_invalid_frame: # 根据 valid_frame_mask 过滤无效帧，剔除无效帧的相机外参、内参以及其他相关信息。
            extrinsics_opencv = extrinsics_opencv[valid_frame_mask]
            intrinsics_opencv = intrinsics_opencv[valid_frame_mask]
            if extra_params is not None:
                extra_params = extra_params[valid_frame_mask]
            invalid_ids = torch.nonzero(~valid_frame_mask).squeeze(1)
            invalid_ids = invalid_ids.cpu().numpy().tolist()
            if len(invalid_ids) > 0:
                for invalid_id in invalid_ids:
                    reconstruction.deregister_image(invalid_id)

        img_size = images.shape[-1]  # H or W, the same for square

        # 如果图像顺序被改变，恢复原始的图像顺序。使用 center_order 调整所有相关张量（如相机参数、轨迹、可见性等）以恢复顺序。
        if center_order is not None:
            # NOTE we changed the image order previously, now we need to scwitch it back
            extrinsics_opencv = extrinsics_opencv[center_order]
            intrinsics_opencv = intrinsics_opencv[center_order]
            if extra_params is not None:
                extra_params = extra_params[center_order]
            pred_track = pred_track[:, center_order]
            pred_vis = pred_vis[:, center_order]
            if pred_score is not None:
                pred_score = pred_score[:, center_order]


        # 恢复图像和相机的原始分辨率。
        if back_to_original_resolution:
            reconstruction = self.rename_colmap_recons_and_rescale_camera(
                reconstruction,
                image_paths,
                crop_params,
                img_size,
                shared_camera=self.cfg.shared_camera,
                shift_point2d_to_original_res=self.cfg.shift_point2d_to_original_res,
            )

            # Also rescale the intrinsics_opencv tensor
            # 将图像名称（name）映射到对应的图像 ID（imgid）
            fname_to_id = {
                reconstruction.images[imgid].name: imgid
                for imgid in reconstruction.images
            }
            intrinsics_original_res = []
            # We assume the returned extri and intri cooresponds to the order of sorted image_paths
            for fname in sorted(image_paths):
                pyimg = reconstruction.images[fname_to_id[fname]]
                pycam = reconstruction.cameras[pyimg.camera_id]
                intrinsics_original_res.append(pycam.calibration_matrix())
            # 遍历排序后的 image_paths，从重建对象中提取每个图像对应的相机内参矩阵。
            # 将提取的内参矩阵逐个添加到列表 intrinsics_original_res 中
            intrinsics_opencv = torch.from_numpy(
                np.stack(intrinsics_original_res)
            ).to(device)
            # torch.from_numpy()：将 NumPy 数组转换为 PyTorch 张量。
            # np.stack(intrinsics_original_res)：
            # 使用 NumPy 的 np.stack() 函数将列表中的多个内参矩阵沿新维度拼接，生成一个形状为 (N, 3, 3) 的数组。

        predictions["extrinsics_opencv"] = extrinsics_opencv
        # NOTE! If not back_to_original_resolution, then intrinsics_opencv
        # cooresponds to the resized one (e.g., 1024x1024)
        predictions["intrinsics_opencv"] = intrinsics_opencv
        predictions["points3D"] = points3D
        predictions["points3D_rgb"] = points3D_rgb
        predictions["reconstruction"] = reconstruction
        predictions["extra_params"] = extra_params
        predictions["unproj_dense_points3D"] = None  # placeholder here
        predictions["valid_2D_mask"] = valid_2D_mask
        predictions["pred_track"] = pred_track
        predictions["pred_vis"] = pred_vis
        predictions["pred_score"] = pred_score
        predictions["valid_tracks"] = valid_tracks
        
        predictions["additional_points_dict"] = additional_points_dict
        
        return predictions

    def triangulate_extra_points(
        self,
        images,
        masks,
        fmaps_for_tracker,
        bound_bboxes,
        intrinsics_opencv,
        extra_params,
        extrinsics_opencv,
        image_paths,
        frame_num,
    ):
        """
        Triangulate extra points for each frame and return a dictionary containing 3D points and their RGB values.

        Returns:
            dict: A dictionary containing 3D points and their RGB values for each frame.
        """
        from vggsfm.models.utils import sample_features4d

        additional_points_dict = {}
        for frame_idx in range(frame_num):
            rect_for_sample = bound_bboxes[:, frame_idx].clone()
            rect_for_sample = rect_for_sample.floor()
            rect_for_sample[:, :2] += self.cfg.extra_pt_pixel_interval // 2
            rect_for_sample[:, 2:] -= self.cfg.extra_pt_pixel_interval // 2
            grid_points = generate_grid_samples(
                rect_for_sample, pixel_interval=self.cfg.extra_pt_pixel_interval
            )
            grid_points = grid_points.floor()

            grid_rgb = sample_features4d(
                images[:, frame_idx], grid_points[None]
            ).squeeze(0)

            if self.cfg.extra_by_neighbor > 0:
                neighbor_start, neighbor_end = sample_subrange(
                    frame_num, frame_idx, self.cfg.extra_by_neighbor
                )
            else:
                neighbor_start = 0
                neighbor_end = frame_num

            rel_frame_idx = frame_idx - neighbor_start

            extra_track, extra_vis, extra_score = predict_tracks(
                self.cfg.query_method,
                self.cfg.max_query_pts,
                self.track_predictor,
                images[:, neighbor_start:neighbor_end],
                (
                    masks[:, neighbor_start:neighbor_end]
                    if masks is not None
                    else masks
                ),
                fmaps_for_tracker[:, neighbor_start:neighbor_end],
                [rel_frame_idx],
                fine_tracking=False,
                bound_bboxes=bound_bboxes[:, neighbor_start:neighbor_end],
                query_points_dict={rel_frame_idx: grid_points[None]},
            )

            extra_params_neighbor = (
                extra_params[neighbor_start:neighbor_end]
                if extra_params is not None
                else None
            )
            extrinsics_neighbor = extrinsics_opencv[neighbor_start:neighbor_end]
            intrinsics_neighbor = intrinsics_opencv[neighbor_start:neighbor_end]

            extra_track_normalized = cam_from_img(
                extra_track, intrinsics_neighbor, extra_params_neighbor
            )

            (extra_triangulated_points, extra_inlier_num, extra_inlier_mask) = (
                triangulate_tracks(
                    extrinsics_neighbor,
                    extra_track_normalized.squeeze(0),
                    track_vis=extra_vis.squeeze(0),
                    track_score=extra_score.squeeze(0),
                )
            )

            valid_triangulation_mask = extra_inlier_num > 3

            valid_poins3D_mask, _ = filter_all_points3D(
                extra_triangulated_points,
                extra_track.squeeze(0),
                extrinsics_neighbor,
                intrinsics_neighbor,
                extra_params=extra_params_neighbor,  # Pass extra_params to filter_all_points3D
                max_reproj_error=self.cfg.max_reproj_error,
            )

            valid_triangulation_mask = torch.logical_and(
                valid_triangulation_mask, valid_poins3D_mask
            )

            extra_points3D = extra_triangulated_points[valid_triangulation_mask]
            extra_points3D_rgb = grid_rgb[valid_triangulation_mask]

            additional_points_dict[image_paths[frame_idx]] = {
                "points3D": extra_points3D,
                "points3D_rgb": extra_points3D_rgb,
                "uv": grid_points[valid_triangulation_mask],
            }

        return additional_points_dict

    def extract_sparse_depth_and_point_from_reconstruction(self, predictions):
        """
        Extracts sparse depth and 3D points from the reconstruction.

        Args:
            predictions (dict): Contains reconstruction data with a 'reconstruction' key.

        Returns:
            dict: Updated predictions with 'sparse_depth' and 'sparse_point' keys.
        """
        reconstruction = predictions["reconstruction"]
        sparse_depth = defaultdict(list)
        sparse_point = defaultdict(list)
        # Extract sparse depths from SfM points
        for point3D_idx in reconstruction.points3D:
            pt3D = reconstruction.points3D[point3D_idx]
            for track_element in pt3D.track.elements:
                pyimg = reconstruction.images[track_element.image_id]
                pycam = reconstruction.cameras[pyimg.camera_id]
                img_name = pyimg.name
                projection = pyimg.cam_from_world * pt3D.xyz
                depth = projection[-1]
                # NOTE: uv here cooresponds to the (x, y)
                # at the original image coordinate
                # instead of the padded&resized one
                uv = pycam.img_from_cam(projection)
                sparse_depth[img_name].append(np.append(uv, depth))
                sparse_point[img_name].append(np.append(pt3D.xyz, point3D_idx))
        predictions["sparse_depth"] = sparse_depth
        predictions["sparse_point"] = sparse_point
        return predictions

    def dense_reconstruct(self, predictions, image_paths, original_images):
        """
        Args:
            predictions (dict): A dictionary containing the sparse reconstruction results.
            image_paths (list): A list of paths to the input images.
            original_images (dict): Dictionary with image basename as keys and original
                numpy images (rgb) as values.

        The function performs the following steps:
        1. Predicts dense depth maps using a monocular depth estimation model, e.g., DepthAnything V2.
        2. Extracts sparse depths from the SfM reconstruction.
        3. Aligns the dense depth maps with the sparse reconstruction.
        4. Updates the predictions dictionary with the dense point cloud data.
        """

        print("Predicting dense depth maps via monocular depth estimation.")

        disp_dict = extract_dense_depth_maps(
            self.depth_model, image_paths, original_images
        )

        sparse_depth = predictions["sparse_depth"]
        reconstruction = predictions["reconstruction"]

        # Align dense depth maps
        print("Aligning dense depth maps by sparse SfM points")
        depth_dict, unproj_dense_points3D = align_dense_depth_maps(
            reconstruction,
            sparse_depth,
            disp_dict,
            original_images,
            visual_dense_point_cloud=self.cfg.visual_dense_point_cloud,
        )

        # Update predictions with dense reconstruction results
        predictions["depth_dict"] = depth_dict
        predictions["unproj_dense_points3D"] = unproj_dense_points3D

        return predictions

    def save_dense_depth_maps(self, depth_dict, output_dir):
        """
        Save the dense depth maps to disk.

        Args:
            depth_dict (dict): Dictionary containing depth maps.
            output_dir (str): Directory to save the depth maps.
        """
        depth_dir = os.path.join(output_dir, "depths")
        os.makedirs(depth_dir, exist_ok=True)
        for img_basename in depth_dict:
            depth_map = depth_dict[img_basename]
            depth_map_path = os.path.join(depth_dir, img_basename)

            name_wo_extension = os.path.splitext(depth_map_path)[0]
            out_fname_with_bin = name_wo_extension + ".bin"
            write_array(depth_map, out_fname_with_bin)

    def make_reprojection_video(
        self, predictions, video_size, image_paths, original_images
    ):
        """
        Create a video with reprojections of the 3D points onto the original images.

        Args:
            predictions (dict): A dictionary containing the reconstruction results,
                                including 3D points and camera parameters.
            video_size (tuple): A tuple specifying the size of the output video (width, height).
            image_paths (list): A list of paths to the input images.
            output_dir (str): The directory to save the output video.
            original_images (dict): Dictionary with image basename as keys and original
                numpy images (rgb) as values.
        """
        reconstruction = predictions["reconstruction"]
        sparse_depth = predictions["sparse_depth"]
        sparse_point = predictions["sparse_point"]

        image_dir_prefix = os.path.dirname(image_paths[0])
        image_paths = [os.path.basename(imgpath) for imgpath in image_paths]

        img_with_circles_list = create_video_with_reprojections(
            image_dir_prefix,
            video_size,
            reconstruction,
            image_paths,
            sparse_depth,
            sparse_point,
            original_images,
        )

        return img_with_circles_list

    def save_reprojection_video(
        self, img_with_circles_list, video_size, output_dir
    ):
        """
        Save the reprojection video to disk.

        Args:
            img_with_circles_list (list): List of images with circles to be included in the video.
            video_size (tuple): A tuple specifying the size of the output video (width, height).
            output_dir (str): The directory to save the output video.
        """
        visual_dir = os.path.join(output_dir, "visuals")
        os.makedirs(visual_dir, exist_ok=True)
        save_video_with_reprojections(
            os.path.join(visual_dir, "reproj.mp4"),
            img_with_circles_list,
            video_size,
        )

    def save_sparse_reconstruction(
        self, predictions, seq_name=None, output_dir=None
    ):
        """
        Save the reconstruction results in COLMAP format.

        Args:
            predictions (dict): Reconstruction results including camera parameters and 3D points.
            seq_name (str, optional): Sequence name for default output directory.
            output_dir (str, optional): Directory to save the reconstruction.

        Saves camera parameters, 3D points, and other data in COLMAP-compatible format.
        """
        # Export prediction as colmap format
        reconstruction_pycolmap = predictions["reconstruction"]
        if output_dir is None:
            output_dir = os.path.join("output", seq_name)

        sfm_output_dir = os.path.join(output_dir, "sparse")
        print("-" * 50)
        print(
            f"The output has been saved in COLMAP style at: {sfm_output_dir} "
        )
        os.makedirs(sfm_output_dir, exist_ok=True)
        reconstruction_pycolmap.write(sfm_output_dir)

    def visualize_3D_in_visdom(
        self, predictions, seq_name=None, output_dir=None
    ):
        """
        This function takes the predictions from the reconstruction process and visualizes
        the 3D point cloud and camera positions in Visdom. It handles both sparse and dense
        reconstructions if available. Requires a running Visdom server and PyTorch3D library.

        Args:
            predictions (dict): Reconstruction results including 3D points and camera parameters.
            seq_name (str, optional): Sequence name for visualization.
            output_dir (str, optional): Directory for saving output files.
        """

        if "points3D_rgb" in predictions:
            pcl = Pointclouds(
                points=predictions["points3D"][None],
                features=predictions["points3D_rgb"][None],
            )
        else:
            pcl = Pointclouds(points=predictions["points3D"][None])

        extrinsics_opencv = predictions["extrinsics_opencv"]

        # From OpenCV/COLMAP to PyTorch3D
        rot_PT3D = extrinsics_opencv[:, :3, :3].clone().permute(0, 2, 1)
        trans_PT3D = extrinsics_opencv[:, :3, 3].clone()
        trans_PT3D[:, :2] *= -1
        rot_PT3D[:, :, :2] *= -1
        visual_cameras = PerspectiveCamerasVisual(
            R=rot_PT3D, T=trans_PT3D, device=trans_PT3D.device
        )

        visual_dict = {"scenes": {"points": pcl, "cameras": visual_cameras}}

        unproj_dense_points3D = predictions["unproj_dense_points3D"]
        if unproj_dense_points3D is not None:
            unprojected_rgb_points_list = []
            for unproj_img_name in sorted(unproj_dense_points3D.keys()):
                unprojected_rgb_points = torch.from_numpy(
                    unproj_dense_points3D[unproj_img_name]
                )
                unprojected_rgb_points_list.append(unprojected_rgb_points)

                # Separate 3D point locations and RGB colors
                point_locations = unprojected_rgb_points[0]  # 3D point location
                rgb_colors = unprojected_rgb_points[1]  # RGB color

                # Create a mask for points within the specified range
                valid_mask = point_locations.abs().max(-1)[0] <= 512

                # Create a Pointclouds object with valid points and their RGB colors
                point_cloud = Pointclouds(
                    points=point_locations[valid_mask][None],
                    features=rgb_colors[valid_mask][None],
                )

                # Add the point cloud to the visual dictionary
                visual_dict["scenes"][f"unproj_{unproj_img_name}"] = point_cloud

        fig = plot_scene(visual_dict, camera_scale=0.05)

        env_name = f"demo_visual_{seq_name}"
        print(f"Visualizing the scene by visdom at env: {env_name}")

        self.viz.plotlyplot(fig, env=env_name, win="3D")

    def visualize_3D_in_gradio(
        self, predictions, seq_name=None, output_dir=None
    ):
        from vggsfm.utils.gradio import (
            vggsfm_predictions_to_glb,
            visualize_by_gradio,
        )

        # Convert predictions to GLB scene
        glbscene = vggsfm_predictions_to_glb(predictions)

        visual_dir = os.path.join(output_dir, "visuals")

        os.makedirs(visual_dir, exist_ok=True)

        sparse_glb_file = os.path.join(visual_dir, "sparse.glb")

        # Export the GLB scene to the specified file
        glbscene.export(file_obj=sparse_glb_file)

        # Visualize the GLB file using Gradio
        visualize_by_gradio(sparse_glb_file)

        unproj_dense_points3D = predictions["unproj_dense_points3D"]
        if unproj_dense_points3D is not None:
            print(
                "Dense point cloud visualization in Gradio is not supported due to time constraints."
            )

    def rename_colmap_recons_and_rescale_camera(
        self,
        reconstruction,
        image_paths,
        crop_params,
        img_size,
        shift_point2d_to_original_res=False,
        shared_camera=False,
    ):
        rescale_camera = True

        for pyimageid in reconstruction.images:
            # Reshaped the padded&resized image to the original size
            # Rename the images to the original names
            pyimage = reconstruction.images[pyimageid]
            pycamera = reconstruction.cameras[pyimage.camera_id]
            pyimage.name = image_paths[pyimageid]

            if rescale_camera:
                # Rescale the camera parameters
                pred_params = copy.deepcopy(pycamera.params)
                real_image_size = crop_params[0, pyimageid][:2]
                resize_ratio = real_image_size.max() / img_size
                real_focal = resize_ratio * pred_params[0]
                real_pp = real_image_size.cpu().numpy() // 2

                pred_params[0] = real_focal
                pred_params[1:3] = real_pp
                pycamera.params = pred_params
                pycamera.width = real_image_size[0]
                pycamera.height = real_image_size[1]

                resize_ratio = resize_ratio.item()

            if shift_point2d_to_original_res:
                # Also shift the point2D to original resolution
                top_left = crop_params[0, pyimageid][-4:-2].abs().cpu().numpy()
                for point2D in pyimage.points2D:
                    point2D.xy = (point2D.xy - top_left) * resize_ratio

            if shared_camera:
                # If shared_camera, all images share the same camera
                # no need to rescale any more
                rescale_camera = False

        return reconstruction


################################################ Helper Functions


def move_to_device(tensor, device):
    return tensor.to(device) if tensor is not None else None


def add_batch_dimension(tensor):
    return tensor.unsqueeze(0) if tensor is not None else None


def predict_tracks(
    query_method,
    max_query_pts, # 一个查询帧最多可以查询的点数
    track_predictor,
    images,
    masks,
    fmaps_for_tracker,
    query_frame_indexes,
    fine_tracking,
    bound_bboxes=None,
    query_points_dict=None,
    max_points_num=163840, # 163840 用于限制所有查询帧加起来 可以查询的特征点数量
):
    """
    Predict tracks for the given images and masks. predict_tracks 函数，它的作用是预测图像序列中的特征点轨迹

    This function predicts the tracks for the given images and masks using the specified query method
    and track predictor. It finds query points, and predicts the tracks, visibility, and scores for the query frames.

    Args:
        query_method (str): The methods to use for querying points (e.g., "sp", "sift", "aliked", or "sp+sift).
        max_query_pts (int): The maximum number of query points.
        track_predictor (object): The track predictor object used for predicting tracks.
        images (torch.Tensor): A tensor of shape (B, T, C, H, W) representing a batch of images.
        masks (torch.Tensor): A tensor of shape (B, T, 1, H, W) representing masks for the images. 1 indicates ignored.
        fmaps_for_tracker (torch.Tensor): A tensor of feature maps for the tracker.
        query_frame_indexes (list): A list of indices representing the query frames.
        fine_tracking (bool): Whether to perform fine tracking.
        bound_bboxes (torch.Tensor, optional): A tensor of shape (B, T, 4) representing bounding boxes for the images.
        max_points_num (int): The maximum number of points to process in one chunk.
                              If the total number of points (T * N) exceeds max_points_num,
                              the query points are split into smaller chunks.
                              Default is 163840, suitable for 40GB GPUs.

    Returns:
        tuple: A tuple containing the predicted tracks, visibility, and scores.
            - pred_track (torch.Tensor): A tensor of shape (B, T, N, 2) representing the predicted tracks.
            - pred_vis (torch.Tensor): A tensor of shape (B, T, N) representing the visibility of the predicted tracks.
            - pred_score (torch.Tensor): A tensor of shape (B, T, N) representing the scores of the predicted tracks.
    """
    pred_track_list = []
    pred_vis_list = []
    pred_score_list = []

    frame_num = images.shape[1]
    device = images.device

    if fmaps_for_tracker is None:
        fmaps_for_tracker = track_predictor.process_images_to_fmaps(images)

    for query_index in query_frame_indexes:# 5 3 4 代码循环遍历这些索引，依次处理每个查询帧。
        print(f"Predicting tracks with query_index = {query_index}")

        if bound_bboxes is not None: # 如果有边界框约束，则仅在当前帧的 该范围内提取特征点。
            bound_bbox = bound_bboxes[:, query_index]
        else:
            bound_bbox = None

        mask = masks[:, query_index] if masks is not None else None

        # Find query_points at the query frame
        if query_points_dict is None:
            query_points = get_query_points(
                images[:, query_index], # 这一帧对应的图片 1 3 1024 1024
                mask,
                query_method,
                max_query_pts,
                bound_bbox=bound_bbox,
            ) # 1 2048 2
        else:
            query_points = query_points_dict[query_index]  # 取此帧对应的查询点 要事先提供

        # Switch so that query_index frame stays at the first frame
        # This largely simplifies the code structure of tracker
        new_order = calculate_index_mappings(
            query_index, frame_num, device=device
        )
        images_feed, fmaps_feed = switch_tensor_order(
            [images, fmaps_for_tracker], new_order
        )

        all_points_num = images_feed.shape[1] * max_query_pts

        if all_points_num > max_points_num: # 为了应对内存不足的问题，当需要处理的点（query_points）的总数超过设定的上限时，将点数据分割成多个小块，分别进行预测，从而降低一次性内存消耗
            print('Predict tracks in chunks to fit in memory')

            # Split query_points into smaller chunks to avoid memory issues
            all_points_num = images_feed.shape[1] * query_points.shape[1]
            
            shuffle_indices = torch.randperm(query_points.size(1)) # 使用 torch.randperm 随机打乱点的顺序。
            query_points = query_points[:, shuffle_indices] # 这种打乱有助于在分块时避免因数据的顺序性可能带来的不均衡问题
            
            num_splits = (all_points_num + max_points_num - 1) // max_points_num # 使用整除运算计算出需要分割成多少块
            fine_pred_track, pred_vis, pred_score = predict_tracks_in_chunks(
                track_predictor,
                images_feed,
                query_points,
                fmaps_feed,
                fine_tracking,
                num_splits,
            )
        else:
            # Feed into track predictor
            fine_pred_track, _, pred_vis, pred_score = track_predictor(
                images_feed, # 排序之后的此查询帧在第一位 1 8 3 1024 1024
                query_points, # 1 2048 2
                fmaps=fmaps_feed, # 排序之后的特征向量
                fine_tracking=fine_tracking,
            )

        # Switch back the predictions 我知道了 这个的意思是换回去 再换一边（不久还回去了吗）太妙了 之前不理解的也懂了
        fine_pred_track, pred_vis, pred_score = switch_tensor_order(
            [fine_pred_track, pred_vis, pred_score], new_order
        )

        # Append predictions for different queries
        pred_track_list.append(fine_pred_track)
        pred_vis_list.append(pred_vis)
        pred_score_list.append(pred_score)

    pred_track = torch.cat(pred_track_list, dim=2)
    pred_vis = torch.cat(pred_vis_list, dim=2)
    pred_score = torch.cat(pred_score_list, dim=2)

    return pred_track, pred_vis, pred_score
 # Final cost

def comple_nonvis_frames(
    query_method,
    max_query_pts,
    track_predictor,
    images,
    masks,
    fmaps_for_tracker,
    preds,
    fine_tracking,
    bound_bboxes=None,
    min_vis=500,
):
    """
    Completes non-visible frames by predicting additional 2D matches.
   补全那些可见性不足（visible inliers 太少）的帧，即对于一些目标在某些帧中没有足够明显的匹配点，该函数会利用额外的查询来预测更多的二维匹配
    This function identifies frames with insufficient visible inliers and uses them as query frames
    to predict additional 2D matches. It iteratively processes these non-visible frames until they
    have enough 2D matches or a final trial is reached.

    Args:
        query_method (str): The methods to use for querying points
                            (e.g., "sp", "sift", "aliked", or "sp+sift).
        max_query_pts (int): The maximum number of query points to use.
        track_predictor (object): The track predictor model.
        images (torch.Tensor): A tensor of shape (B, T, C, H, W) representing a batch of images.
        masks (torch.Tensor): A tensor of shape (B, T, 1, H, W) representing masks for the images.
        fmaps_for_tracker (torch.Tensor): Feature maps for the tracker.
        preds (tuple): A tuple containing predicted tracks, visibility, and scores.
        fine_tracking (bool): Whether to perform fine tracking.
        bound_bboxes (torch.Tensor, optional): Bounding boxes for the images.
        min_vis (int, optional): The minimum number of visible inliers required. Default is 500.
    Returns:
        tuple: A tuple containing updated predicted tracks, visibility, and scores.
    """
    pred_track, pred_vis, pred_score = preds
    # if a frame has too few visible inlier, use it as a query
    non_vis_frames = ( # 0 7
        torch.nonzero((pred_vis.squeeze(0) > 0.05).sum(-1) < min_vis) # 去掉批次维度1 大于 0.05 认为是可见的 对最后一个维度求和，即统计每一帧（或每个目标）中可见的内点数量
        .squeeze(-1) # 与预设的最低可见数量 min_vis（例如 500）比较，标记出可见点不足的帧 torch.nonzero(...)：返回这些条件满足的位置
        .tolist() # .squeeze(-1).tolist()：压缩多余维度并转换为 Python 列表，得到所有“非可见”（可见匹配不足）的帧的索引列表
    )
    last_query = -1 # last_query 用于记录上一次选择作为查询的帧的索引，初始设为 -1。
    final_trial = False # 标志是否已经进入最后一次尝试（例如采用更严格的查询方法和减少查询点数

    while len(non_vis_frames) > 0:
        print("Processing non visible frames: ", non_vis_frames)

        if non_vis_frames[0] == last_query: # 如果当前列表中第一个非可见帧（non_vis_frames[0]）与上一次处理的帧相同，说明上次补全尝试后，该帧依然匹配不足
            print("The non visible frame still does not has enough 2D matches")
            final_trial = True # 设置 final_trial 为 True，表示进入最后一次尝试
            query_method = "sp+sift+aliked" # 将 query_method 修改为 "sp+sift+aliked"（结合多种查询方法以增强匹配效果
            max_query_pts = max_query_pts // 2 # 同时将 max_query_pts 减半，以降低查询点数量（可能为了减少冗余或加快计算
            non_vis_query_list = non_vis_frames # 并把所有非可见帧都作为查询输入（non_vis_query_list = non_vis_frames
        else:
            non_vis_query_list = [non_vis_frames[0]] # 如果不是重复同一帧，则选取列表中第一个（也就是自己）作为查询帧，

        last_query = non_vis_frames[0] # 最后更新 last_query 为当前查询的帧索引：
        pred_track_comple, pred_vis_comple, pred_score_comple = predict_tracks(# 这一行调用 predict_tracks 函数
            query_method, # ，传入当前的查询方法、最大查询点数、跟踪预测器、图像、掩码、特征图、待处理的非可见帧列表、是否进行细粒度跟踪以及边界框信息
            max_query_pts,
            track_predictor,
            images,
            masks,
            fmaps_for_tracker,
            non_vis_query_list, # 以这个当作查询帧，再次进行轨迹查询
            fine_tracking,
            bound_bboxes,
        ) # 返回的结果是补全后的预测轨迹（pred_track_comple）、可见性（pred_vis_comple）和匹配得分（pred_score_comple）

        # concat predictions
        pred_track = torch.cat([pred_track, pred_track_comple], dim=2) # +2048=8192
        pred_vis = torch.cat([pred_vis, pred_vis_comple], dim=2)
        pred_score = torch.cat([pred_score, pred_score_comple], dim=2)
        non_vis_frames = (
            torch.nonzero((pred_vis.squeeze(0) > 0.05).sum(-1) < min_vis)
            .squeeze(-1)
            .tolist()
        ) # 这段代码会重新检查每一帧中可见匹配点的数量 因为里面的值都变了

        if final_trial:
            break
    return pred_track, pred_vis, pred_score


def predict_tracks_in_chunks(
    track_predictor,
    images_feed,
    query_points,
    fmaps_feed,
    fine_tracking,
    num_splits,
):
    """
    Process query points in smaller chunks to avoid memory issues.

    Args:
        track_predictor (object): The track predictor object used for predicting tracks.
        images_feed (torch.Tensor): A tensor of shape (B, T, C, H, W) representing a batch of images.
        query_points (torch.Tensor): A tensor of shape (B, N, 2) representing the query points.
        fmaps_feed (torch.Tensor): A tensor of feature maps for the tracker.
        fine_tracking (bool): Whether to perform fine tracking.
        num_splits (int): The number of chunks to split the query points into.

    Returns:
        tuple: A tuple containing the concatenated predicted tracks, visibility, and scores.
    """
    split_query_points = torch.chunk(query_points, num_splits, dim=1)

    fine_pred_track_list = []
    pred_vis_list = []
    pred_score_list = []

    for split_points in split_query_points:
        # Feed into track predictor for each split
        fine_pred_track, _, pred_vis, pred_score = track_predictor(
            images_feed,
            split_points,
            fmaps=fmaps_feed,
            fine_tracking=fine_tracking,
        )
        fine_pred_track_list.append(fine_pred_track)
        pred_vis_list.append(pred_vis)
        pred_score_list.append(pred_score)

    # Concatenate the results from all splits
    fine_pred_track = torch.cat(fine_pred_track_list, dim=2)
    pred_vis = torch.cat(pred_vis_list, dim=2)
    if pred_score is not None:
        pred_score = torch.cat(pred_score_list, dim=2)
    else:
        pred_score = None

    return fine_pred_track, pred_vis, pred_score


def get_query_points(
    query_image,
    seg_invalid_mask,
    query_method,
    max_query_num=4096,
    det_thres=0.005,# det_thres 控制特征点检测的阈值，值越大，检测到的点越少
    bound_bbox=None,
):
    """
    Extract query points from the given query image using the specified method.

    This function extracts query points from the given query image using the specified method(s).
    It supports multiple methods such as "sp" (SuperPoint), "sift" (SIFT), and "aliked" (ALIKED).
    The function also handles invalid masks and bounding boxes to filter out unwanted regions.

    Args:
        query_image (torch.Tensor): A tensor of shape (B, C, H, W) representing the query image.
        seg_invalid_mask (torch.Tensor, optional):
                        A tensor of shape (B, 1, H, W) representing the segmentation invalid mask.
        query_method (str): The method(s) to use for extracting query points
                            (e.g., "sp", "sift", "aliked", or combinations like "sp+sift").
        max_query_num (int, optional): The maximum number of query points to extract. Default is 4096.
        det_thres (float, optional): The detection threshold for keypoint extraction. Default is 0.005.
        bound_bbox (torch.Tensor, optional): A tensor of shape (B, 4) representing bounding boxes for the images.

    Returns:
        torch.Tensor: A tensor of shape (B, N, 2) representing the extracted query points,
                        where N is the number of query points.
    """

    methods = query_method.split("+") # split("+") 允许同时使用多个方法，比如 ["sp", "sift"]
    pred_points = []

    for method in methods: # 根据 query_method 选择不同的特征点提取器
        if "sp" in method:
            extractor = SuperPoint(
                max_num_keypoints=max_query_num, detection_threshold=det_thres
            )
        elif "sift" in method:
            extractor = SIFT(max_num_keypoints=max_query_num)
        elif "aliked" in method:
            extractor = ALIKED(
                max_num_keypoints=max_query_num, detection_threshold=det_thres
            )
        else:
            raise NotImplementedError(
                f"query method {method} is not supprted now"
            )
        extractor = extractor.cuda().eval() # 提高计算效率，避免训练模式带来的不稳定性。
        invalid_mask = None

        if bound_bbox is not None:
            x_min, y_min, x_max, y_max = map(int, bound_bbox[0]) # 第一组坐标转换为整数 逐注意 前面我都将其变为正数了 这保证了坐标值不会为负，适用于图像像素坐标的要求
            bbox_valid_mask = torch.zeros_like( # 取出第一通道得到 (B, H, W) 的掩码 1 1024 1024
                query_image[:, 0], dtype=torch.bool
            )
            bbox_valid_mask[:, y_min:y_max, x_min:x_max] = 1 # 在掩码中，将边界框内的区域标记为 True
            invalid_mask = ~bbox_valid_mask # 1 1024 1024对有效区域掩码取反，即边界框内为 False（有效），边界框外为 True（无效）。

        if seg_invalid_mask is not None: # 某个位置只要在边界框外或在分割无效区域内，就会被标记为无效（True）。
            seg_invalid_mask = seg_invalid_mask.squeeze(1).bool() # 使用 squeeze(1) 将其转换为 (B, H, W) 的布尔型张量。
            invalid_mask = (  # 1 1024 1024 如果之前没有通过 bound_bbox 生成 invalid_mask（即 invalid_mask 为 None），则直接使用 seg_invalid_mask。
                seg_invalid_mask #
                if invalid_mask is None
                else torch.logical_or(invalid_mask, seg_invalid_mask) # 如果已有 invalid_mask（来自边界框），则通过 torch.logical_or 将两者合并
            )

        query_points = extractor.extract(
            query_image, invalid_mask=invalid_mask # 1 3 1024 1024 只用提供帧和掩码1 1024 1024 人家有人叫自己的计算方式 这是大模型
        )["keypoints"] # 结果是（B=1，querypoint_num=2048,2为有效坐标）
        pred_points.append(query_points) # 装了不同查询方式查询点的 列表

    query_points = torch.cat(pred_points, dim=1) # 按照这些方式的点数 那一维进行连接 1 2048 2

    if query_points.shape[1] > max_query_num:
        random_point_indices = torch.randperm(query_points.shape[1])[
            :max_query_num # [:max_query_num] 切片取前 max_query_num 个随机索引，保证选出的点数量不超过限制。
        ] # torch.randperm(query_points.shape[1]) 生成一个长度为 N 的随机排列，即包含从 0 到 N-1 的所有索引的随机顺序。
        query_points = query_points[:, random_point_indices, :] # 这里使用随机索引对原始 query_points 进行索引，得到一个新的张量。 小于max值

    return query_points
