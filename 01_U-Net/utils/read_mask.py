import os
import numpy as np
from PIL import Image

# 1. 填入你的具体路径
img_path = (
    "/home/monkey/下载/U-net数据集/DIC-C2DH-HeLa_train/DIC-C2DH-HeLa/01/t002.tif"
)
mask_path = "/home/monkey/下载/U-net数据集/DIC-C2DH-HeLa_train/DIC-C2DH-HeLa/01_GT/SEG/man_seg002.tif"

# 2. 读取图片
img = Image.open(img_path)
mask = Image.open(mask_path)

# 3. 转换成 numpy 数组看尺寸和数值
img_arr = np.array(img)
mask_arr = np.array(mask)

print("原图尺寸:", img_arr.shape, "数据类型:", img_arr.dtype)
print("标签尺寸:", mask_arr.shape, "数据类型:", mask_arr.dtype)
print(
    "标签里包含的唯一数值 (unique):", np.unique(mask_arr)
)  # 看看是不是只有 0 和 其它数字