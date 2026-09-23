# 导入系统路径模块
import os

# 导入绘图库 Matplotlib，用来拼图显示效果
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 导入 PyTorch 相关库
import torch
import torchvision.transforms.functional as TF
from U_net import UNet


def predict_and_visualize():
    # 1. 设置计算设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 实例化模型结构并加载刚才训练好的权重
    model = UNet().to(device)
    weights_path = "unet_cell.pth"

    # map_location 确保权重能正确加载到当前设备上
    model.load_state_dict(torch.load(weights_path, map_location=device))
    # eval() 将模型切换为评估/测试模式（固定 BatchNorm 的均值和方差）
    model.eval()

    # 3. 指定一张测试图像和对应的真实标签
    img_path = "./data/DIC-C2DH-HeLa/01/t002.tif"
    mask_path = "./data/DIC-C2DH-HeLa/01_GT/SEG/man_seg002.tif"

    # 读取原图并归一化
    raw_img = Image.open(img_path)
    img_arr = np.array(raw_img, dtype=np.float32) / 255.0
    # 构造网络输入张量: [Batch=1, Channel=1, H=512, W=512]
    img_tensor = torch.from_numpy(img_arr).unsqueeze(0).unsqueeze(0).to(device)

    # 读取标签并进行中心裁剪 (裁出与模型输出匹配的 324x324)
    raw_mask = Image.open(mask_path)
    mask_arr = (np.array(raw_mask) > 0).astype(np.uint8)
    mask_tensor = torch.from_numpy(mask_arr)
    # 中心裁剪
    cropped_gt = TF.center_crop(mask_tensor, [324, 324]).numpy()

    # 同样把原图也中心裁剪出 324x324，方便展示时与预测结果严格对齐
    cropped_img = TF.center_crop(torch.from_numpy(img_arr), [324, 324]).numpy()

    # 4. 前向推理（预测）
    # torch.no_grad() 告诉 PyTorch 不需要计算梯度，节省显存并加快运行速度
    with torch.no_grad():
        output = model(img_tensor)  # 输出形状: [1, 2, 324, 324]
        # output 在通道维度上有两个得分：通道 0（背景）与 通道 1（细胞）
        # torch.argmax(dim=1) 会挑选得分最大的那个类别下标作为最终分类（输出 0 或 1）
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    # 5. 画图展示：1行3列
    plt.figure(figsize=(12, 4))

    # 子图 1：原始显微镜图像（裁剪对齐区域）
    plt.subplot(1, 3, 1)
    plt.title("Original Image (Crop 324x324)")
    plt.imshow(cropped_img, cmap="gray")
    plt.axis("off")

    # 子图 2：真实人工标注标签 (Ground Truth)
    plt.subplot(1, 3, 2)
    plt.title("Ground Truth (Cell Mask)")
    plt.imshow(cropped_gt, cmap="gray")
    plt.axis("off")

    # 子图 3：U-Net 模型的预测结果
    plt.subplot(1, 3, 3)
    plt.title("U-Net Prediction")
    plt.imshow(pred_mask, cmap="gray")
    plt.axis("off")

    # 保存对比图片
    save_result_path = "result_comparison.png"
    plt.tight_layout()
    plt.savefig(save_result_path, dpi=300)
    print(f">>> 预测对比图已保存至: {save_result_path}")
    plt.show()


if __name__ == "__main__":
    predict_and_visualize()