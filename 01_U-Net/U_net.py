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
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=0
            ),
            nn.BatchNorm2d(out_channels),
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

class DecoderBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()

        self.up=nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )

        self.conv=DoubleConv(
            out_channels*2,
            out_channels
        )

    def forward(self,x,skip):
        x=self.up(x)
        diff_h=skip.size(2)-x.size(2)
        diff_w=skip.size(3)-x.size(3)

        skip=skip[
            :,
            :,
            diff_h//2:diff_h//2+x.size(2),
            diff_w//2:diff_w//2+x.size(3)
        ]

        x=torch.cat([x,skip],dim=1)
        x=self.conv(x)
        return x
class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        #Encoder部分的搭建
        self.enc1=EncoderBlock(1,64)
        self.enc2=EncoderBlock(64,128)
        self.enc3=EncoderBlock(128,256)
        self.enc4=EncoderBlock(256,512)

        self.bottleneck=DoubleConv(512,1024)

        #进行Decoder部分的搭建
        self.dec1=DecoderBlock(1024,512)
        self.dec2=DecoderBlock(512,256)
        self.dec3=DecoderBlock(256,128)
        self.dec4=DecoderBlock(128,64)

        self.final=nn.Conv2d(64,2,kernel_size=1)

    def forward(self,x):
        #保存的不是 Pool 后的东西，而是 Pool 前的 skip
        x1,skip1=self.enc1(x)
        x2,skip2=self.enc2(x1)
        x3,skip3=self.enc3(x2)
        x4,skip4=self.enc4(x3)

        x=self.bottleneck(x4)

        x=self.dec1(x,skip4)
        x=self.dec2(x,skip3)
        x=self.dec3(x,skip2)
        x=self.dec4(x,skip1)

        x=self.final(x)

        return x


if __name__ == "__main__":
    # x=torch.randn(1,1024,28,28)
    # skip=torch.randn(1,512,64,64)
    # block=DecoderBlock(1024,512)
    # y=block(x,skip)
    # print(y.shape)
    #model=DoubleConv(1,64)

    #y=model(x)
    #block=EncoderBlock(1,64)
    #x,skip=block(x)
    # print(x.shape)
    # print(skip.shape)
    model=UNet()
    x=torch.randn(1,1,572,572)
    y=model(x)

    # print(f"input:{x.shape}")
    # print(f"output:{y.shape}")
