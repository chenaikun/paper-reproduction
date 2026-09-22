# 导入系统路径模块与文件匹配工具
import glob
import os

# 导入图像处理库 PIL 与数值计算库 numpy
import numpy as np
from PIL import Image

# 导入 PyTorch 相关模块
import torch
import torchvision.transforms.functional as TF  # 提供张量裁剪等几何操作的工具
from torch.utils.data import DataLoader, Dataset


class HeLaDataset(Dataset):

    def __init__(self, data_root, target_size=324):
        # target_size 是模型最终输出的尺寸，原版 U-Net 输入 512 时输出为 324
        self.target_size = target_size
        self.pairs = []

        # 遍历 01 与 02 文件夹
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

        # 1. 读取原图并归一化到 [0, 1]
        image = Image.open(img_path)
        img_arr = np.array(image, dtype=np.float32) / 255.0
        # 增加通道维度变成 [1, 512, 512]
        img_tensor = torch.from_numpy(img_arr).unsqueeze(0)

        # 2. 读取掩膜标签
        mask = Image.open(mask_path)
        mask_arr = np.array(mask)
        # 将大于 0 的像素作为细胞前景，转成整数标签（0 代表背景，1 代表细胞）
        mask_arr = (mask_arr > 0).astype(np.int64)
        mask_tensor = torch.from_numpy(mask_arr)  # 尺寸为 [512, 512]

        # 3. 对标签执行中心裁剪，匹配模型输出的 [324, 324]
        # center_crop 会从 512x512 的正中心切出 324x324 的区域
        mask_tensor = TF.center_crop(mask_tensor, [self.target_size, self.target_size])

        # CrossEntropyLoss 要求目标标签形状为 [Batch, H, W]，数据类型为 long
        return img_tensor, mask_tensor.long()


if __name__ == "__main__":
    # 测试裁剪是否正常工作
    data_root = "./data/DIC-C2DH-HeLa"
    dataset = HeLaDataset(data_root, target_size=324)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    imgs, masks = next(iter(dataloader))
    print("原图输入形状:", imgs.shape)  # 应为 [2, 1, 512, 512]
    print(
        "裁剪后标签形状:", masks.shape
    )  # 应为 [2, 324, 324]，与模型输出的空间尺寸完全一致