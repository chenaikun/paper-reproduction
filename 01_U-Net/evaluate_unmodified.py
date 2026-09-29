from dataset_unmodified import UnmodifiedSegmentationDataset
from U_net_baseline import UNet
import torch
from torch.utils.data import DataLoader
import numpy as np
dataset = UnmodifiedSegmentationDataset(
    data_root="./data/unmodified-data",
    split="test",
    seed=42,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet().to(device)

model.load_state_dict(
    torch.load(
        "results/unmodified_baseline_ce_20epoch/best_model.pth",
        map_location=device,
    )
)

model.eval()

loader = DataLoader(dataset, batch_size=1, shuffle=False)


#保存交并比，dice,前景占比
iou_list = []
dice_list = []
fg_list = []

with torch.no_grad():
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)

        pred = predictions[0].bool()
        target = masks[0].bool()

        intersection = (pred & target).sum().item()
        union = (pred | target).sum().item()

        iou = intersection / union if union > 0 else 1.0

        pred_count = pred.sum().item()
        target_count = target.sum().item()

        dice = (
            2 * intersection / (pred_count + target_count)
            if pred_count + target_count > 0
            else 1.0
        )

        iou_list.append(iou)
        dice_list.append(dice)
        fg_list.append(pred.float().mean().item())


print(f"测试集数量: {len(dataset)}")
print(f"Mean IoU: {np.mean(iou_list) * 100:.2f}%")
print(f"Std IoU: {np.std(iou_list) * 100:.2f}%")
print(f"Mean Dice: {np.mean(dice_list) * 100:.2f}%")
print(f"Std Dice: {np.std(dice_list) * 100:.2f}%")
print(f"Mean foreground ratio: {np.mean(fg_list):.4f}")