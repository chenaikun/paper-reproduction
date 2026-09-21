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

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1=EncoderBlock(1,64)
        self.enc2=EncoderBlock(64,128)
        self.enc3=EncoderBlock(128,256)
        self.enc4=EncoderBlock(256,512)

        self.bottleneck=DoubleConv(512,1024)


    def forward(self,x):
        #保存的不是 Pool 后的东西，而是 Pool 前的 skip
        x1,skip1=self.enc1(x)
        x2,skip2=self.enc2(x1)
        x3,skip3=self.enc3(x2)
        x4,skip4=self.enc4(x3)

        x=self.bottleneck(x4)




if __name__ == "__main__":
    x=torch.randn(1,1,572,572)
    #model=DoubleConv(1,64)

    #y=model(x)
    block=EncoderBlock(1,64)
    x,skip=block(x)
    # print(x.shape)
    # print(skip.shape)

