import os

from dataset_original import HeLaDataset
from losses_original import WeightedCrossEntropyLoss

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from U_net_original import UNet


def train():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"当前使用的训练设备是：{device}")

    dataset = HeLaDataset(
        data_root="./data/DIC-C2DH-HeLa",
        target_size=324,
        is_train=False
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True
    )

    model = UNet().to(device)

    # U-Net boundary weighted loss
    criterion = WeightedCrossEntropyLoss(
        w0=10.0,
        sigma=5.0
    )

    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.99
    )

    num_epochs = 60

    print(">>> 开始训练...")

    for epoch in range(num_epochs):

        model.train()

        total_loss = 0.0

        for step, (
            images,
            masks,
            instance_masks
        ) in enumerate(dataloader):

            images = images.to(device)
            masks = masks.to(device)
            instance_masks = instance_masks.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                masks,
                instance_masks
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        avg_loss = (
            total_loss /
            len(dataloader)
        )

        print(
            f"Epoch "
            f"[{epoch + 1:02d}/{num_epochs:02d}]"
            f"-平均损失：{avg_loss:.4f}"
        )

    save_path = (
        "results/original_weighted/"
        "unet_original_weighted_ce.pth"
    )

    os.makedirs(
        os.path.dirname(save_path),
        exist_ok=True
    )

    torch.save(
        model.state_dict(),
        save_path
    )

    print(
        f">>> 训练完成！"
        f"模型权重已保存至：{save_path}"
    )


if __name__ == "__main__":
    train()