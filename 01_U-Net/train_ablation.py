"""
统一消融训练脚本：从 Baseline 逐步逼近 U-Net 论文原方案

使用方法：
  方式1 — 命令行参数（推荐，配合 run_ablation.sh）：
    python train_ablation.py --name step0_baseline_adam_ce_dice --model baseline --optimizer adam --loss ce_dice

  方式2 — 修改顶部 CONFIG 字典：
    python train_ablation.py

逐步逼近路径：
  Step 0: Baseline (BN + Adam + CE+Dice + flip/rot)     — 起始点
  Step 1: Baseline (BN + Adam + CE + flip/rot)           — 去 Dice
  Step 2: Baseline (BN + SGD + CE + flip/rot)            — 换 SGD
  Step 3: Baseline (BN + SGD + WeightedCE + flip/rot)    — 加边界权重
  Step 4: Original (无BN + SGD + WeightedCE + flip/rot)  — 去 BN
  Step 5: Original (无BN + SGD + WeightedCE + elastic)   — 弹性变形
"""

import os
import sys
import time
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# ============================================================
# 默认配置（可被命令行参数覆盖）
# ============================================================
DEFAULT_CONFIG = {
    "name": "step0_baseline_adam_ce_dice",
    "dataset": "hela",
    "data_root": "./data/DIC-C2DH-HeLa",
    "init": "default",
    "model": "baseline",
    "optimizer": "adam",
    "loss": "ce_dice",
    "lr": 5e-4,
    "momentum": 0.99,
    "w0": 10.0,
    "sigma": 5.0,
    "elastic_deform": False,
    "epochs": 60,
    "batch_size": 1,
    "grad_clip": 0.0,
    "seed": 42,
}
# ============================================================

#工厂模式
def build_model(cfg):
    """根据配置构建模型"""
    if cfg["model"] == "baseline":
        from U_net_baseline import UNet
        return UNet()
    elif cfg["model"] == "original":
        from U_net_original import UNet
        return UNet()
    elif cfg["model"] == "no_skip":
        from U_net_no_skip import UNetNoSkip
        return UNetNoSkip()
    else:
        raise ValueError(f"未知模型: {cfg['model']}")

#工厂模式
def build_criterion(cfg, device):
    """根据配置构建损失函数"""
    if cfg["loss"] == "ce_dice":
        from losses_Dice import CombinedLoss
        return CombinedLoss(), True  # (criterion, needs_instance_mask)
    elif cfg["loss"] == "ce":
        return nn.CrossEntropyLoss(), False
    elif cfg["loss"] == "weighted_ce":
        from losses_original import WeightedCrossEntropyLoss
        return WeightedCrossEntropyLoss(
            w0=cfg["w0"], sigma=cfg["sigma"]
        ), True
    else:
        raise ValueError(f"未知损失: {cfg['loss']}")

#工厂模式
def build_optimizer(cfg, model):
    """根据配置构建优化器"""
    if cfg["optimizer"] == "adam":
        return optim.Adam(model.parameters(), lr=cfg["lr"])
    elif cfg["optimizer"] == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=cfg["lr"],
            momentum=cfg["momentum"],
        )
    else:
        raise ValueError(f"未知优化器: {cfg['optimizer']}")


def initialize_model(model, init_type):
    """Apply the requested convolution initialization before optimization."""
    if init_type == "default":
        return
    if init_type != "he":
        raise ValueError(f"未知初始化方式: {init_type}")

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            # He/Kaiming normal initialization: std = sqrt(2 / fan_in).
            nn.init.kaiming_normal_(
                module.weight,
                mode="fan_in",
                nonlinearity="relu",
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)

#计算交并比
def calculate_iou(pred_mask, true_mask, smooth=1e-6):
    """计算单样本 IoU"""
    pred = pred_mask.view(-1).bool()
    target = true_mask.view(-1).bool()
    intersection = (pred & target).sum().float().item()
    union = (pred | target).sum().float().item()
    return (intersection + smooth) / (union + smooth)


@torch.no_grad()
def evaluate_on_set(model, dataloader, device):
    """
    在给定数据集上评估，返回:
      mean_iou, mean_dice, mean_fg_ratio
    """
    model.eval()
    iou_list = []
    dice_list = []
    fg_list = []

    for batch in dataloader:
        if len(batch) == 3:
            images, masks, _ = batch
        else:
            images, masks = batch

        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        for j in range(preds.shape[0]):
            iou = calculate_iou(preds[j], masks[j])
            intersection = (preds[j].view(-1).bool() & masks[j].view(-1).bool()).sum().float().item()
            total = preds[j].float().sum().item() + masks[j].float().sum().item()
            dice = (2.0 * intersection + 1e-6) / (total + 1e-6)

            iou_list.append(iou)
            dice_list.append(dice)
            fg_list.append(preds[j].float().mean().item())

    return np.mean(iou_list), np.mean(dice_list), np.mean(fg_list)


def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 设置随机种子
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["seed"])

    # 确定是否需要 instance_mask
    needs_instance = cfg["loss"] == "weighted_ce"

    # 构建数据集。默认仍使用旧 HeLa 数据；新数据通过 --dataset unmodified 选择。
    if cfg["dataset"] == "hela":
        from dataset_unified import HeLaDataset

        train_dataset = HeLaDataset(
            data_root=cfg["data_root"],
            target_size=324,
            is_train=True,
            split="train",
            return_instance_mask=needs_instance,
            elastic_deform=cfg["elastic_deform"],
        )
        val_dataset = HeLaDataset(
            data_root=cfg["data_root"],
            target_size=324,
            is_train=False,
            split="val",
            return_instance_mask=needs_instance,
        )
    elif cfg["dataset"] == "unmodified":
        if needs_instance:
            raise ValueError(
                "unmodified 数据集只有二值标签，暂不支持 weighted_ce；"
                "请先使用 ce 或 ce_dice。"
            )
        from dataset_unmodified import UnmodifiedSegmentationDataset

        train_dataset = UnmodifiedSegmentationDataset(
            data_root=cfg["data_root"],
            split="train",
            target_size=324,
            seed=cfg["seed"],
        )
        val_dataset = UnmodifiedSegmentationDataset(
            data_root=cfg["data_root"],
            split="val",
            target_size=324,
            seed=cfg["seed"],
        )
    else:
        raise ValueError(f"未知数据集: {cfg['dataset']}")

    train_loader = DataLoader(
        train_dataset, batch_size=cfg["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False
    )

    # 构建模型
    model = build_model(cfg).to(device)
    initialize_model(model, cfg["init"])
    criterion, criterion_needs_instance = build_criterion(cfg, device)
    optimizer = build_optimizer(cfg, model)

    # 输出目录
    save_dir = f"results/{cfg['name']}"
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "log.txt")

    # 打印配置
    print("=" * 60)
    print(f"实验: {cfg['name']}")
    print(f"数据集: {cfg['dataset']} ({cfg['data_root']})")
    print(f"初始化: {cfg['init']}")
    print(f"模型: {cfg['model']} | 优化器: {cfg['optimizer']} | 损失: {cfg['loss']}")
    print(f"lr: {cfg['lr']} | epochs: {cfg['epochs']} | batch_size: {cfg['batch_size']}")
    print(f"弹性变形: {cfg['elastic_deform']} | 梯度裁剪: {cfg['grad_clip']}")
    print(f"训练集: {len(train_dataset)} 张 | 验证集: {len(val_dataset)} 张")
    print(f"设备: {device}")
    print("=" * 60)

    # 训练循环
    best_val_iou = 0.0
    log_lines = []

    for epoch in range(cfg["epochs"]):
        model.train()
        total_loss = 0.0
        epoch_start = time.time()

        for batch in train_loader:
            if len(batch) == 3:
                images, masks, instance_masks = batch
                instance_masks = instance_masks.to(device)
            else:
                images, masks = batch
                instance_masks = None

            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            if criterion_needs_instance and instance_masks is not None:
                loss = criterion(outputs, masks, instance_masks)
            else:
                loss = criterion(outputs, masks)

            loss.backward()

            # 梯度裁剪
            if cfg["grad_clip"] > 0:
                nn.utils.clip_grad_norm_(
                    model.parameters(), cfg["grad_clip"]
                )

            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # 验证集评估
        val_iou, val_dice, val_fg = evaluate_on_set(model, val_loader, device)

        elapsed = time.time() - epoch_start

        # 退化检测
        degenerate_flag = " ⚠️ 退化！" if val_fg > 0.95 else ""

        # 保存最佳模型
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_path = os.path.join(save_dir, "best_model.pth")
            torch.save(model.state_dict(), best_path)

        # 日志行
        log_line = (
            f"Epoch [{epoch + 1:02d}/{cfg['epochs']:02d}] "
            f"- loss: {avg_loss:.4f} "
            f"- val_IoU: {val_iou * 100:.2f}% "
            f"- val_Dice: {val_dice * 100:.2f}% "
            f"- fg_ratio: {val_fg:.4f}"
            f"{degenerate_flag} "
            f"- best_IoU: {best_val_iou * 100:.2f}% "
            f"({elapsed:.1f}s)"
        )
        print(log_line)
        log_lines.append(log_line)

        # 如果模型退化，尽早警告
        if val_fg > 0.95:
            print(
                f"  ⚠️ 警告：验证集预测前景比例 {val_fg:.4f} > 0.95，"
                f"模型可能退化为全前景预测！"
            )

    # 保存最终模型
    final_path = os.path.join(save_dir, "final_model.pth")
    torch.save(model.state_dict(), final_path)

    # 写日志
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"实验配置: {cfg}\n\n")
        for line in log_lines:
            f.write(line + "\n")

    # 最终评估
    final_iou, final_dice, final_fg = evaluate_on_set(model, val_loader, device)
    print("\n" + "=" * 60)
    print(f"训练完成！最终验证集指标:")
    print(f"  Mean IoU : {final_iou * 100:.2f}%")
    print(f"  Mean Dice: {final_dice * 100:.2f}%")
    print(f"  前景比例 : {final_fg:.4f}")
    print(f"  最佳 IoU : {best_val_iou * 100:.2f}%")
    print(f"模型保存至: {save_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="U-Net 消融训练")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None, choices=["hela", "unmodified"])
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--init", type=str, default=None, choices=["default", "he"])
    parser.add_argument("--model", type=str, default=None, choices=["baseline", "original", "no_skip"])
    parser.add_argument("--optimizer", type=str, default=None, choices=["adam", "sgd"])
    parser.add_argument("--loss", type=str, default=None, choices=["ce_dice", "ce", "weighted_ce"])
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--w0", type=float, default=None)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--elastic-deform", action="store_true", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    # 从默认配置开始，用命令行参数覆盖
    cfg = dict(DEFAULT_CONFIG)
    arg_map = {
        "name": args.name,
        "dataset": args.dataset,
        "data_root": args.data_root,
        "init": args.init,
        "model": args.model,
        "optimizer": args.optimizer,
        "loss": args.loss,
        "lr": args.lr,
        "momentum": args.momentum,
        "w0": args.w0,
        "sigma": args.sigma,
        "elastic_deform": args.elastic_deform,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_clip": args.grad_clip,
        "seed": args.seed,
    }
    for k, v in arg_map.items():
        if v is not None:
            cfg[k] = v

    train(cfg)
