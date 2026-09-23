import torch
from dataset import HeLaDataset
from torch.utils.data import DataLoader
from U_net import UNet

# 1. 检查数据与标签
dataset = HeLaDataset(data_root="./data/DIC-C2DH-HeLa", target_size=324)
loader = DataLoader(dataset, batch_size=2, shuffle=False)

imgs, masks = next(iter(loader))
print(f"原图数值范围: min={imgs.min():.4f}, max={imgs.max():.4f}")
print(f"裁剪后标签中的唯一值: {torch.unique(masks).tolist()}")
print(
    f"裁剪后标签中前景(细胞)像素所占比例: {(masks == 1).float().mean().item() * 100:.2f}%"
)

# 2. 检查单步反向传播的梯度
model = UNet()
criterion = torch.nn.CrossEntropyLoss()
out = model(imgs)
loss = criterion(out, masks)
loss.backward()

# 检查第一层和最后一层的梯度大小
first_conv_grad = model.enc1.conv.double_conv[0].weight.grad
final_conv_grad = model.final.weight.grad

print(
    f"最后一层(final)权重的平均梯度幅值: {final_conv_grad.abs().mean().item():.6f}"
)
print(
    f"第一层(enc1)权重的平均梯度幅值: {first_conv_grad.abs().mean().item():.6f}"
)