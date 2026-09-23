import os

from dataset_baseline import HeLaDataset


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from U_net_baseline import UNet
from losses_Dice import CombinedLoss

def train():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的训练设备是：{device}")

    dataset=HeLaDataset(data_root="./data/DIC-C2DH-HeLa",target_size=324)
    dataloader=DataLoader(dataset,batch_size=2,shuffle=True)


    model=UNet().to(device)

    # criterion=nn.CrossEntropyLoss()

    #class_weights=torch.tensor([1.0,5.0]).to(device)
    #criterion=nn.CrossEntropyLoss(weight=class_weights)
    criterion = CombinedLoss()
    #使用经典的Adam优化器
    optimizer=optim.Adam(model.parameters(),lr=5e-4)

    num_epochs=60

    print(">>> 开始训练...")

    for epoch in range(num_epochs):
        model.train()
        total_loss=0.0

        for step,(images,masks) in enumerate(dataloader):
            images=images.to(device)
            masks=masks.to(device)

            #梯度清零
            optimizer.zero_grad()
            outputs=model(images)

            loss=criterion(outputs,masks)
            loss.backward()#反向传播

            total_loss+=loss.item()
            optimizer.step()
        avg_loss=total_loss/len(dataloader)
        print(f"Epoch [{epoch+1:02d}/{num_epochs:02d}]-平均损失：{avg_loss:.4f}")

    #保存训练好的网络权重
    save_path="unet_cell.pth"
    torch.save(model.state_dict(),save_path)
    print(f">>> 训练完成！模型权重已保存至：{save_path}")


if __name__ == "__main__":
    train()
