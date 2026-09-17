import torch
import torch.nn as nn


class SpatialProjection(nn.Module):

    def __init__(self,
                 input_dim=256,
                 output_channels=256,
                 spatial_size=7):

        super().__init__()

        self.output_channels = output_channels
        self.spatial_size = spatial_size

        self.project = nn.Sequential(
            nn.Linear(
                input_dim,
                output_channels * spatial_size * spatial_size
            ),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        x = self.project(x)

        x = x.view(
            x.size(0),
            self.output_channels,
            self.spatial_size,
            self.spatial_size
        )

        return x