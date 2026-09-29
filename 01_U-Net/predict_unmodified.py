"""Visualize predictions on the downloaded U-Net PNG dataset."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as TF

from dataset_unmodified import UnmodifiedSegmentationDataset
from U_net_baseline import UNet


def calculate_metrics(prediction, target):
    pred = prediction.bool()
    true = target.bool()
    intersection = (pred & true).sum().item()
    union = (pred | true).sum().item()
    total = pred.sum().item() + true.sum().item()

    iou = intersection / union if union else 1.0
    dice = 2.0 * intersection / total if total else 1.0
    return iou, dice


def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = UnmodifiedSegmentationDataset(
        data_root=args.data_root,
        split=args.split,
        seed=args.seed,
    )
    if not 0 <= args.index < len(dataset):
        raise IndexError(
            f"index={args.index} 超出 {args.split} 集范围 0~{len(dataset) - 1}"
        )

    image, target = dataset[args.index]

    model = UNet().to(device)
    state_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        logits = model(image.unsqueeze(0).to(device))
        prediction = logits.argmax(dim=1).squeeze(0).cpu()

    iou, dice = calculate_metrics(prediction, target)
    target_fg = target.float().mean().item()
    pred_fg = prediction.float().mean().item()

    cropped_image = TF.center_crop(image, [324, 324]).squeeze(0)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(cropped_image.numpy(), cmap="gray")
    axes[0].set_title("Input image (center crop)")

    axes[1].imshow(target.numpy(), cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"Ground truth\nforeground={target_fg:.3f}")

    axes[2].imshow(prediction.numpy(), cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(
        f"Prediction\nIoU={iou:.3f}, Dice={dice:.3f}, foreground={pred_fg:.3f}"
    )

    for axis in axes:
        axis.axis("off")

    figure.suptitle(
        f"Unmodified dataset: {args.split} sample {args.index}",
        fontsize=12,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    print(f"设备: {device}")
    print(f"数据: {args.split}[{args.index}] / 共 {len(dataset)} 张")
    print(f"权重: {args.weights}")
    print(f"IoU: {iou * 100:.2f}%")
    print(f"Dice: {dice * 100:.2f}%")
    print(f"真实前景比例: {target_fg:.4f}")
    print(f"预测前景比例: {pred_fg:.4f}")
    print(f"对比图: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="新 U-Net 数据集预测可视化")
    parser.add_argument(
        "--data-root",
        default="./data/unmodified-data",
    )
    parser.add_argument(
        "--weights",
        default="results/unmodified_baseline_ce_20epoch/best_model.pth",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="results/unmodified_baseline_ce_20epoch/prediction_val_00.png",
    )
    predict(parser.parse_args())
