import torch.nn as nn


class PredictionHead(nn.Module):

    def __init__(self, feature_dim=256, output_dim=24):

        super().__init__()

        self.classifier = nn.Sequential(

            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),

            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(inplace=True),

            nn.Dropout(0.2),

            nn.Linear(64, output_dim)
        )

    def forward(self, x):

        return self.classifier(x)