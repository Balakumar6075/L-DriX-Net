import torch
import torch.nn as nn


class AdaptiveFusion(nn.Module):

    def __init__(self):

        super().__init__()

        # Project gaze feature (64 → 256)
        self.gaze_projection = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(inplace=True)
        )

        # Fuse face + scene + gaze
        self.fusion = nn.Sequential(

            nn.Linear(256 + 256 + 256, 512),

            nn.BatchNorm1d(512),

            nn.ReLU(inplace=True),

            nn.Dropout(0.3),

            nn.Linear(512, 256),

            nn.ReLU(inplace=True)
        )

    def forward(self, face, scene, gaze):

        gaze = self.gaze_projection(gaze)

        fused = torch.cat([face, scene, gaze], dim=1)

        fused = self.fusion(fused)

        return fused