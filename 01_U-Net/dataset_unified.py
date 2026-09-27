"""
统一数据集：合并 dataset_baseline 和 dataset_original 的功能
- 支持 train/val 划分 (Seq01 训练, Seq02 验证)
- 可选返回 instance_mask (用于加权损失)
- 可选弹性变形增强 (U-Net 论文的核心数据增强)
"""

import os
import glob
import random
import numpy as np
from PIL import Image

import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

from scipy.ndimage import gaussian_filter


class HeLaDataset(Dataset):
    def __init__(
        self,
        data_root,
        target_size=324,
        is_train=True,
        split="all",
        return_instance_mask=False,
        elastic_deform=False,
    ):
        """
        参数:
            data_root: 数据根目录
            target_size: 掩膜中心裁剪尺寸
            is_train: 是否启用数据增强 (翻转/旋转/弹性变形)
            split: "train" = Seq01, "val" = Seq02, "all" = 两个序列
            return_instance_mask: 为 True 时返回 (img, binary_mask, instance_mask)
            elastic_deform: 为 True 且 is_train=True 时添加弹性变形
        """
        self.target_size = target_size
        self.is_train = is_train
        self.return_instance_mask = return_instance_mask
        self.elastic_deform = elastic_deform and is_train

        # 根据 split 选择序列
        if split == "train":
            sequences = ["01"]
        elif split == "val":
            sequences = ["02"]
        else:
            sequences = ["01", "02"]

        self.pairs = []

        for seq in sequences:
            seg_dir = os.path.join(data_root, f"{seq}_GT", "SEG")
            seg_files = sorted(
                glob.glob(os.path.join(seg_dir, "man_seg*.tif"))
            )

            for seg_path in seg_files:
                filename = os.path.basename(seg_path)
                num_str = filename.replace("man_seg", "").replace(".tif", "")
                img_path = os.path.join(data_root, seq, f"t{num_str}.tif")

                if os.path.exists(img_path):
                    self.pairs.append((img_path, seg_path))

    def __len__(self):
        return len(self.pairs)

    def _elastic_transform(self, image, mask, alpha=40, sigma=6):
        """
        弹性变形：U-Net 论文的核心数据增强方法
        在规则网格上施加随机位移，用高斯平滑得到连续位移场，
        然后对图像和掩膜做相同的双线性/最近邻插值变换。

        参数:
            alpha: 位移幅度 (论文建议 ~40 像素)
            sigma: 位移场平滑的高斯 sigma (论文建议 ~6 像素)
        """
        if isinstance(image, Image.Image):
            img_arr = np.array(image, dtype=np.float32)
        else:
            img_arr = np.array(image, dtype=np.float32)

        if isinstance(mask, Image.Image):
            mask_arr = np.array(mask, dtype=np.int64)
        else:
            mask_arr = np.array(mask, dtype=np.int64)

        shape = img_arr.shape[:2]  # (H, W)

        # 生成随机位移场
        dx = gaussian_filter(
            (np.random.rand(*shape) * 2 - 1), sigma
        ) * alpha
        dy = gaussian_filter(
            (np.random.rand(*shape) * 2 - 1), sigma
        ) * alpha

        # 构建采样网格
        y, x = np.meshgrid(
            np.arange(shape[0]),
            np.arange(shape[1]),
            indexing="ij",
        )

        indices_x = np.clip(
            (x + dx).astype(np.int32), 0, shape[1] - 1
        )
        indices_y = np.clip(
            (y + dy).astype(np.int32), 0, shape[0] - 1
        )

        # 对图像做弹性变换
        img_transformed = img_arr[indices_y, indices_x]

        # 对掩膜做同样的变换
        mask_transformed = mask_arr[indices_y, indices_x]

        return img_transformed, mask_transformed

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        image = Image.open(img_path)
        mask = Image.open(mask_path)

        # ========================
        # 1. 数据增强
        # ========================
        if self.is_train:

            # 50% 随机水平翻转
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # 50% 随机垂直翻转
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            # 50% 随机小角度旋转
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                image = TF.rotate(image, angle)
                mask = TF.rotate(
                    mask,
                    angle,
                    interpolation=TF.InterpolationMode.NEAREST,
                )

            # 弹性变形 (U-Net 论文的核心增强)
            if self.elastic_deform and random.random() > 0.5:
                img_arr, mask_arr = self._elastic_transform(
                    image, mask
                )
                image = Image.fromarray(
                    img_arr.astype(np.uint8)
                )
                mask = Image.fromarray(
                    mask_arr.astype(np.int32).clip(0, None)
                )

        # ========================
        # 2. image -> Tensor
        # ========================
        img_arr = np.array(image, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_arr).unsqueeze(0)

        # ========================
        # 3. 处理掩膜
        # ========================
        instance_arr = np.array(mask, dtype=np.int64)
        binary_arr = (instance_arr > 0).astype(np.int64)

        instance_mask = torch.from_numpy(instance_arr)
        binary_mask = torch.from_numpy(binary_arr)

        # 中心裁剪到 target_size
        instance_mask = TF.center_crop(
            instance_mask, [self.target_size, self.target_size]
        )
        binary_mask = TF.center_crop(
            binary_mask, [self.target_size, self.target_size]
        )

        # ========================
        # 4. 返回
        # ========================
        if self.return_instance_mask:
            return (
                img_tensor,
                binary_mask.long(),
                instance_mask.long(),
            )
        else:
            return img_tensor, binary_mask.long()
