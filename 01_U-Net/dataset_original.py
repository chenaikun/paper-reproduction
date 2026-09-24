import os
import glob
import random
import numpy as np
from PIL import Image

import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset


class HeLaDataset(Dataset):
    def __init__(self, data_root, target_size=324, is_train=False):
        self.target_size = target_size
        self.is_train = is_train
        self.pairs = []

        for seq in ["01", "02"]:
            seg_dir = os.path.join(data_root, f"{seq}_GT", "SEG")
            seg_files = sorted(
                glob.glob(os.path.join(seg_dir, "man_seg*.tif"))
            )

            for seg_path in seg_files:
                filename = os.path.basename(seg_path)
                num_str = filename.replace("man_seg", "").replace(".tif", "")

                img_path = os.path.join(
                    data_root,
                    seq,
                    f"t{num_str}.tif"
                )

                if os.path.exists(img_path):
                    self.pairs.append((img_path, seg_path))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        image = Image.open(img_path)
        mask = Image.open(mask_path)

        # =========================
        # 1. 数据增强
        # =========================
        if self.is_train:

            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            if random.random() > 0.5:
                angle = random.uniform(-15, 15)

                image = TF.rotate(
                    image,
                    angle
                )

                mask = TF.rotate(
                    mask,
                    angle,
                    interpolation=TF.InterpolationMode.NEAREST
                )

        # =========================
        # 2. image -> Tensor
        # =========================
        img_arr = np.array(
            image,
            dtype=np.float32
        ) / 255.0

        img_tensor = torch.from_numpy(
            img_arr
        ).unsqueeze(0)

        # =========================
        # 3. 保留 instance mask
        # =========================
        instance_arr = np.array(
            mask,
            dtype=np.int64
        )

        instance_mask = torch.from_numpy(
            instance_arr
        )

        # =========================
        # 4. instance mask -> binary mask
        # =========================
        binary_arr = (
            instance_arr > 0
        ).astype(np.int64)

        binary_mask = torch.from_numpy(
            binary_arr
        )

        # =========================
        # 5. 中心裁剪
        # =========================
        instance_mask = TF.center_crop(
            instance_mask,
            [self.target_size, self.target_size]
        )

        binary_mask = TF.center_crop(
            binary_mask,
            [self.target_size, self.target_size]
        )

        return (
            img_tensor,
            binary_mask.long(),
            instance_mask.long()
        )