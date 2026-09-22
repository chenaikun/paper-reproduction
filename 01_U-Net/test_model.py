import torch

from U_net import UNet


#实例化模型

#in_channels=1
#out_channels=2 一般为2

model = UNet()

fake_input=torch.randn(1,1,512,512)

output=model(fake_input)

#打印张量
print(f"fake_input:{fake_input.shape}")
print(f"output:{output.shape}")