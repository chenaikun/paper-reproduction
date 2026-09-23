import os
import glob
import random
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader

class HeLaDataset(Dataset):
    def __init__(self, data_root, target_size=324, is_train=False):
        self.target_size = target_size
        self.is_train = is_train  # 区分是训练还是测试，测试时不加随机增强
        self.pairs = []

        for seq in ["01", "02"]:
            seg_dir = os.path.join(data_root, f"{seq}_GT", "SEG")
            seg_files = sorted(glob.glob(os.path.join(seg_dir, "man_seg*.tif")))
            for seg_path in seg_files:
                filename = os.path.basename(seg_path)
                num_str = filename.replace("man_seg", "").replace(".tif", "")
                img_path = os.path.join(data_root, seq, f"t{num_str}.tif")
                if os.path.exists(img_path):
                    self.pairs.append((img_path, seg_path))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        # 1. 打开 PIL 图像
        image = Image.open(img_path)
        mask = Image.open(mask_path)

        # 2. 如果是训练阶段，执行同步数据增强（原图和掩膜做同样的变换）
        if self.is_train:
            # 50% 概率随机水平翻转
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # 50% 概率随机垂直翻转
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            # 随机小角度微旋 (-15度 到 15度)
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                image = TF.rotate(image, angle)
                # 掩膜旋转使用近邻插值，防止引入非整数类别
                mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

        # 3. 原图转为 Tensor 并归一化 [1, 512, 512]
        img_arr = np.array(image, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_arr).unsqueeze(0)

        # 4. 掩膜转为二值类别标签并中心裁剪 [324, 324]
        mask_arr = (np.array(mask) > 0).astype(np.int64)
        mask_tensor = torch.from_numpy(mask_arr)
        mask_tensor = TF.center_crop(mask_tensor, [self.target_size, self.target_size])

        return img_tensor, mask_tensor.long()