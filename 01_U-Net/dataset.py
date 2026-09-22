# 导入系统路径和文件搜索模块
import glob  # 用于匹配符合规则的文件路径（比如找所有 .tif 图片）
import os  # 用于处理操作系统文件路径（比如拼接文件夹名字）
# 导入图像处理库 PIL（Python Imaging Library）
from PIL import Image  # 用来把磁盘上的图片读取到内存中
# 导入矩阵运算库 NumPy
import numpy as np  # 用于把图片转成多维数组（数字矩阵）进行数学操作
# 导入深度学习框架 PyTorch
import torch  # PyTorch 核心库
from torch.utils.data import (  # Dataset 是数据集模板，DataLoader 是批量加载器
    DataLoader,
    Dataset,
)


# 自定义一个属于我们自己的数据集类，必须继承 PyTorch 提供的 Dataset 类
class HeLaDataset(Dataset):

    # __init__ 是初始化函数：当你创建这个类的时候，它会自动最先执行
    def __init__(self, data_root):
        # self.pairs 用来存放所有的“原图路径”与“标签路径”配对元组
        self.pairs = []

        # 数据集包含 01 和 02 两个子文件夹，我们用循环分别去里面找
        for seq in ["01", "02"]:
            # 拼接出存放标签的文件夹路径，例如：data_root/01_GT/SEG
            seg_dir = os.path.join(data_root, f"{seq}_GT", "SEG")

            # 找到 seg_dir 文件夹下所有以 man_seg 开头、.tif 结尾的文件
            # sorted() 用来按字母顺序从小到大排序
            seg_files = sorted(glob.glob(os.path.join(seg_dir, "man_seg*.tif")))

            # 遍历每一个找到的标签文件
            for seg_path in seg_files:
                # 获取文件名本身（去除前面的长路径），例如得到 "man_seg002.tif"
                filename = os.path.basename(seg_path)

                # 把 "man_seg" 和 ".tif" 替换剔除掉，只留下数字编号，例如得到 "002"
                num_str = filename.replace("man_seg", "").replace(".tif", "")

                # 根据编号拼出对应的原图文件名并组成完整路径，例如：data_root/01/t002.tif
                img_path = os.path.join(data_root, seq, f"t{num_str}.tif")

                # 安全检查：只有原图真实存在于磁盘上时，才把它们成对存起来
                if os.path.exists(img_path):
                    # 将这一对路径作为一个元组添加到列表中
                    self.pairs.append((img_path, seg_path))

    # __len__ 函数：告诉 PyTorch 我们的数据集到底一共有多少张图
    def __len__(self):
        # 返回配对列表的长度（也就是一共有多少对可用数据）
        return len(self.pairs)

    # __getitem__ 函数：告诉 PyTorch 当模型需要第 idx 张图时，应该如何读取和处理
    def __getitem__(self, idx):
        # 1. 根据索引数字 idx，取出对应的一组路径
        img_path, mask_path = self.pairs[idx]

        # 2. 读取原图
        image = Image.open(img_path)  # 用 PIL 打开图片
        img_arr = np.array(
            image, dtype=np.float32
        )  # 转成 numpy 浮点数数组，尺寸为 (512, 512)
        img_arr = (
            img_arr / 255.0
        )  # 归一化：把像素值从 0~255 压缩到 0.0~1.0 之间，方便网络计算

        # 3. 将原图转成 PyTorch 张量（Tensor）
        img_tensor = torch.from_numpy(img_arr)  # 转换为 PyTorch 张量
        # unsqueeze(0) 在最前面插入一个通道维度，形状从 (512, 512) 变为 (1, 512, 512)
        # 1 代表它是单通道灰度图（如果是彩色 RGB 则是 3）
        img_tensor = img_tensor.unsqueeze(0)

        # 4. 读取标签（Mask）
        mask = Image.open(mask_path)  # 打开标签图片
        mask_arr = np.array(mask)  # 转成 numpy 数组

        # 5. 处理标签数值：
        # 原标签里 0 是背景，1、2、3...是不同的细胞
        # (mask_arr > 0) 会把大于 0 的像素变成 True，等于 0 的变成 False
        # .astype(np.float32) 把 True 转成 1.0，False 转成 0.0，实现二值化分类
        mask_arr = (mask_arr > 0).astype(np.float32)

        # 6. 将标签转成 PyTorch 张量，同样增加通道维度，变成 (1, 512, 512)
        mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0)

        # 返回处理好的图像和标签张量给模型
        return img_tensor, mask_tensor


# 下面的代码只在直接运行该脚本（python dataset.py）时执行，被其他文件导入时不会执行
if __name__ == "__main__":
    # 数据集的根目录绝对路径
    data_root = (
        "./data/DIC-C2DH-HeLa"
    )

    # 实例化我们写好的数据集对象
    dataset = HeLaDataset(data_root)
    print(f"数据集加载成功，总样本数: {len(dataset)}")

    # DataLoader 是 PyTorch 的数据加载器，它负责把数据打包成一批一批（Batch）送进显卡
    # batch_size=2 表示一次性打包 2 张图片一起送给模型
    # shuffle=True 表示每轮训练打乱图片顺序，避免模型死记硬背顺序
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 从 DataLoader 里面抽取一组批次数据测试一下
    # iter() 把加载器变成迭代器，next() 抓取第一个 Batch
    sample_imgs, sample_masks = next(iter(dataloader))

    # 打印测试抽取的张量维度
    # 正常期望形状为 [Batch大小, 通道数, 高度, 宽度] -> [2, 1, 512, 512]
    print(
        "抽取的图片批次形状 (Batch, Channel, Height, Width):",
        sample_imgs.shape,
    )
    print(
        "抽取的标签批次形状 (Batch, Channel, Height, Width):",
        sample_masks.shape,
    )