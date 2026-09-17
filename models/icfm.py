import torch
import torch.nn as nn


class ICFM(nn.Module):
    """
    Information Cross Fusion Module
    """

    def __init__(self, feature_dim=256):

        super().__init__()

        # Face projection
        self.face_fc = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True)
        )

        # Scene projection
        self.scene_fc = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True)
        )

        # Cross attention
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=4,
            batch_first=True
        )

        # Output fusion
        self.output = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, face_feature, scene_feature):

        face = self.face_fc(face_feature)
        scene = self.scene_fc(scene_feature)

        # (B,256) -> (B,1,256)
        face = face.unsqueeze(1)
        scene = scene.unsqueeze(1)

        # Cross Attention
        attended, _ = self.attention(
            query=face,
            key=scene,
            value=scene
        )

        attended = attended.squeeze(1)

        fusion = torch.cat(
            [
                face_feature,
                attended
            ],
            dim=1
        )

        fusion = self.output(fusion)

        return fusion