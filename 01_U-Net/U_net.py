import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()

        self.double_conv=nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=0
            ),
            nn.ReLU()
        )

    def forward(self,x):
        return self.double_conv(x)

class EncoderBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()

        self.conv=DoubleConv(in_channels,out_channels)
        self.pool=nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

    def forward(self,x):
        x=self.conv(x)
        skip=x
        x=self.pool(x)

        return x,skip

class DecoderBlock:
    pass

class UNet:
    pass




if __name__ == "__main__":
    x=torch.randn(1,1,572,572)
    #model=DoubleConv(1,64)

    #y=model(x)
    block=EncoderBlock(1,64)
    x,skip=block(x)
    print(x.shape)
    print(skip.shape)