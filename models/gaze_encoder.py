import torch.nn as nn


class GazeEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Linear(24, 64),

            nn.BatchNorm1d(64),

            nn.ReLU(inplace=True),

            nn.Dropout(0.3),

            nn.Linear(64, 64),

            nn.ReLU(inplace=True)

        )

    def forward(self, x):

        return self.encoder(x)