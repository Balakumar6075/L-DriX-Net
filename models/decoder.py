import torch
import torch.nn as nn


class DecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True)

        )

    def forward(self, x):

        return self.block(x)


class AttentionDecoder(nn.Module):

    def __init__(self, in_channels=256):

        super().__init__()

        self.decoder = nn.Sequential(

            # 7 → 14
            DecoderBlock(256, 256),

            # 14 → 28
            DecoderBlock(256, 128),

            # 28 → 56
            DecoderBlock(128, 64),

            # 56 → 112
            DecoderBlock(64, 32),

            # 112 → 224
            DecoderBlock(32, 16),

            nn.Conv2d(
                16,
                1,
                kernel_size=3,
                padding=1
            ),

            nn.Sigmoid()

        )

    def forward(self, x):

        return self.decoder(x)