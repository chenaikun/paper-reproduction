"""
统一评估脚本：支持所有实验的模型架构和 checkpoint

用法:
    python evaluate.py --experiment baseline
    python evaluate.py --experiment no_skip
    python evaluate.py --experiment step0_baseline_adam_ce_dice
    python evaluate.py --experiment step5_original_sgd_weighted_elastic

如果不传 --experiment，默认评估 step0。
"""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader


# 实验名 → (模型导入路径, checkpoint文件名, 是否需要instance_mask)
EXPERIMENT_REGISTRY = {
    # 原始实验 (旧 checkpoint)
    "baseline": {
        "model": "U_net_baseline:UNet",
        "ckpt": "results/baseline/unet_cell_train_baseline.pth",
        "needs_instance": False,
    },
    "no_skip": {
        "model": "U_net_no_skip:UNetNoSkip",
        "ckpt": "results/baseline_no_skip/unet_no_skip.pth",
        "needs_instance": False,
    },
    "original_ce": {
        "model": "U_net_original:UNet",
        "ckpt": "results/original/unet_cell_OriginalArchitecture_SGD_CE.pth",
        "needs_instance": False,
    },
    "original_weighted": {
        "model": "U_net_original:UNet",
        "ckpt": "results/original_weighted/unet_original_weighted_ce.pth",
        "needs_instance": True,
    },
}


def resolve_model(experiment_name):
    """
    解析实验名，返回 (model_instance, ckpt_path, needs_instance_mask)
    对于 step* 开头的消融实验，自动推断模型架构。
    """
    # 先查精确注册表
    if experiment_name in EXPERIMENT_REGISTRY:
        info = EXPERIMENT_REGISTRY[experiment_name]
    else:
        # 消融实验: 从实验名推断
        # step0, step1, step2, step3 → baseline (BN)
        # step4, step5 → original (无BN)
        # no_skip → no_skip
        ckpt_dir = f"results/{experiment_name}"
        ckpt_path = os.path.join(ckpt_dir, "best_model.pth")

        # 如果 best_model.pth 不存在，尝试 final_model.pth
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(ckpt_dir, "final_model.pth")

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"找不到 checkpoint: {ckpt_dir}/best_model.pth 或 final_model.pth"
            )

        # 从实验名推断模型类型
        if "original" in experiment_name or "step4" in experiment_name or "step5" in experiment_name:
            model_key = "U_net_original:UNet"
        elif "no_skip" in experiment_name:
            model_key = "U_net_no_skip:UNetNoSkip"
        else:
            model_key = "U_net_baseline:UNet"

        # 从实验名推断是否需要 instance_mask
        needs_instance = "weighted" in experiment_name

        info = {
            "model": model_key,
            "ckpt": ckpt_path,
            "needs_instance": needs_instance,
        }

    # 实例化模型
    module_name, class_name = info["model"].split(":")
    module = __import__(module_name)
    model_class = getattr(module, class_name)
    model = model_class()

    return model, info["ckpt"], info["needs_instance"]


def calculate_metrics(pred_mask, true_mask, smooth=1e-6):
    """计算单个样本的 IoU 和 Dice"""
    pred = pred_mask.view(-1).bool()
    target = true_mask.view(-1).bool()

    intersection = (pred & target).sum().float().item()
    union = (pred | target).sum().float().item()

    iou = (intersection + smooth) / (union + smooth)
    dice = (2.0 * intersection + smooth) / (
        pred.sum().float().item() + target.sum().float().item() + smooth
    )

    return iou, dice


def evaluate(experiment_name, split="val"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"评估运行设备: {device}")

    # 解析模型和 checkpoint
    model, ckpt_path, needs_instance = resolve_model(experiment_name)
    model = model.to(device)

    print(f"实验: {experiment_name}")
    print(f"模型: {model.__class__.__name__}")
    print(f"Checkpoint: {ckpt_path}")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # 加载数据集 (使用统一数据集)
    from dataset_unified import HeLaDataset

    dataset = HeLaDataset(
        data_root="./data/DIC-C2DH-HeLa",
        target_size=324,
        is_train=False,
        split=split,
        return_instance_mask=needs_instance,
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    iou_list = []
    dice_list = []
    fg_list = []

    print(f"\n>>> 评估 {split} 集 ({len(dataset)} 帧)...")

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            if len(batch) == 3:
                images, masks, _ = batch
            else:
                images, masks = batch

            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            pred_masks = torch.argmax(outputs, dim=1)

            iou, dice = calculate_metrics(pred_masks[0], masks[0])
            fg_ratio = pred_masks[0].float().mean().item()

            iou_list.append(iou)
            dice_list.append(dice)
            fg_list.append(fg_ratio)

            print(
                f"  [{idx + 1:02d}/{len(dataset):02d}] "
                f"IoU: {iou * 100:.2f}% | Dice: {dice * 100:.2f}% | fg: {fg_ratio:.4f}"
            )

    mean_iou = np.mean(iou_list)
    mean_dice = np.mean(dice_list)
    mean_fg = np.mean(fg_list)

    print("-" * 55)
    print(f"实验: {experiment_name} | 评估集: {split}")
    print(f"Mean IoU : {mean_iou * 100:.2f}%")
    print(f"Mean Dice: {mean_dice * 100:.2f}%")
    print(f"Mean fg  : {mean_fg:.4f}")
    print("-" * 55)

    return mean_iou, mean_dice


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="U-Net 评估")
    parser.add_argument(
        "--experiment",
        type=str,
        default="step0_baseline_adam_ce_dice",
        help="实验名称 (如 baseline, no_skip, step0, step1, ...)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "all"],
        help="评估哪个数据集划分",
    )
    args = parser.parse_args()

    evaluate(args.experiment, args.split)
