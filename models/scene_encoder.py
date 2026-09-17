import torch
import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class SceneEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.stage1 = ConvBlock(3, 32)

        self.stage2 = nn.Sequential(
            ConvBlock(32, 64, stride=2),
            ConvBlock(64, 64)
        )

        self.stage3 = nn.Sequential(
            ConvBlock(64, 128, stride=2),
            ConvBlock(128, 128)
        )

        self.stage4 = nn.Sequential(
            ConvBlock(128, 256, stride=2),
            ConvBlock(256, 256)
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.pool(x)

        x = torch.flatten(x, 1)

        return x