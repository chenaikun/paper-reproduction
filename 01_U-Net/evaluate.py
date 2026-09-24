import os
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from dataset_original import HeLaDataset
from U_net_original import UNet

def calculate_metrics(pred_mask, true_mask, smooth=1e-6):
    """
    计算单个样本的 IoU 和 Dice 分数
    pred_mask: 模型二值预测图 (0 或 1)
    true_mask: 真实二值标签图 (0 或 1)
    """
    # 转为扁平的一维布尔向量方便计算集合操作
    pred = pred_mask.view(-1).bool()
    target = true_mask.view(-1).bool()

    # 计算交集 (True Positive)
    intersection = (pred & target).sum().float().item()
    # 计算并集
    union = (pred | target).sum().float().item()

    # IoU = 交集 / 并集
    iou = (intersection + smooth) / (union + smooth)

    # Dice = 2 * 交集 / (预测总数 + 真实总数)
    dice = (2.0 * intersection + smooth) / (pred.sum().float().item() + target.sum().float().item() + smooth)

    return iou, dice

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"评估运行设备: {device}")

    # 1. 载入数据集与模型
    data_root = "./data/DIC-C2DH-HeLa"
    dataset = HeLaDataset(data_root=data_root, target_size=324,is_train=False)
    # batch_size=1 逐张精确统计
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = UNet().to(device)
    model.load_state_dict(torch.load("./results/original_weighted/unet_original_weighted_ce.pth", map_location=device))
    model.eval()

    iou_list = []
    dice_list = []

    print(f">>> 开始评估共有 {len(dataset)} 张样本的数据集...")

    with torch.no_grad():
        for idx, (images, masks) in enumerate(dataloader):
            images = images.to(device)
            masks = masks.to(device)

            # 前向推理
            outputs = model(images)
            # 取通道得分最高的一类作为预测结果 (0 或 1)
            pred_masks = torch.argmax(outputs, dim=1)

            # 计算当前样本的指标
            iou, dice = calculate_metrics(pred_masks[0], masks[0])
            iou_list.append(iou)
            dice_list.append(dice)

            print(f"样本 [{idx+1:02d}/{len(dataset):02d}] - IoU: {iou*100:.2f}%, Dice: {dice*100:.2f}%")

    # 计算整体平均值
    mean_iou = np.mean(iou_list)
    mean_dice = np.mean(dice_list)

    print("-" * 45)
    print(f"全集平均指标:")
    print(f"Mean IoU : {mean_iou * 100:.2f}%")
    print(f"Mean Dice: {mean_dice * 100:.2f}%")
    print("-" * 45)

if __name__ == "__main__":
    evaluate()