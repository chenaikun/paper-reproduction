import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.double_conv = nn.Sequential(
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

    def forward(self, x):
        return self.double_conv(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = DoubleConv(
            in_channels,
            out_channels
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

    def forward(self, x):

        x = self.conv(x)

        x = self.pool(x)

        return x


class DecoderBlockNoSkip(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )

        # 没有 Skip，所以输入只有 out_channels
        self.conv = DoubleConv(
            out_channels,
            out_channels
        )

    def forward(self, x):

        x = self.up(x)

        x = self.conv(x)

        return x


class UNetNoSkip(nn.Module):

    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = EncoderBlock(1, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)

        # Bottleneck
        self.bottleneck = DoubleConv(
            512,
            1024
        )

        # Decoder
        self.dec1 = DecoderBlockNoSkip(
            1024,
            512
        )

        self.dec2 = DecoderBlockNoSkip(
            512,
            256
        )

        self.dec3 = DecoderBlockNoSkip(
            256,
            128
        )

        self.dec4 = DecoderBlockNoSkip(
            128,
            64
        )

        # Output
        self.final = nn.Conv2d(
            64,
            2,
            kernel_size=1
        )

    def forward(self, x):

        x = self.enc1(x)

        x = self.enc2(x)

        x = self.enc3(x)

        x = self.enc4(x)

        x = self.bottleneck(x)

        x = self.dec1(x)

        x = self.dec2(x)

        x = self.dec3(x)

        x = self.dec4(x)

        x = self.final(x)

        return x