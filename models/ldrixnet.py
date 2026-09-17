import torch
import torch.nn as nn

from models.lfem import LFEM
from models.scene_encoder import SceneEncoder
from models.gaze_encoder import GazeEncoder
from models.icfm import ICFM
from models.temporal_attention import TemporalAttention
from models.heads import PredictionHead
from models.decoder import AttentionDecoder
from models.spatial_projection import SpatialProjection


class LDriXNet(nn.Module):

    def __init__(self):

        super().__init__()

        # -----------------------------
        # Face Branch
        # -----------------------------
        self.face_encoder = LFEM()

        # -----------------------------
        # Scene Branch
        # -----------------------------
        self.scene_encoder = SceneEncoder()

        # -----------------------------
        # Gaze Branch
        # -----------------------------
        self.gaze_encoder = GazeEncoder()

        self.gaze_projection = nn.Linear(64, 256)

        # -----------------------------
        # Cross Fusion
        # -----------------------------
        self.icfm = ICFM(feature_dim=256)

        # -----------------------------
        # Temporal Attention
        # -----------------------------
        self.temporal = TemporalAttention()

        # -----------------------------
        # Prediction Head
        # -----------------------------
        self.head = PredictionHead(
            feature_dim=256,
            output_dim=24
        )

        # -----------------------------
        # NEW Spatial Projection
        # -----------------------------
        self.spatial_projection = SpatialProjection(
            input_dim=256,
            output_channels=256,
            spatial_size=7
        )

        # -----------------------------
        # Heatmap Decoder
        # -----------------------------
        self.decoder = AttentionDecoder()

    def forward(self, face, scene, gaze):

        # -----------------------------
        # Feature Extraction
        # -----------------------------
        face_feature = self.face_encoder(face)

        scene_feature = self.scene_encoder(scene)

        gaze_feature = self.gaze_encoder(gaze)

        gaze_feature = self.gaze_projection(gaze_feature)

        # -----------------------------
        # Fusion
        # -----------------------------
        fused = self.icfm(
            face_feature,
            scene_feature
        )

        fused = fused + gaze_feature

        # -----------------------------
        # Temporal Attention
        # -----------------------------
        temporal_input = fused.unsqueeze(1)

        temporal_feature, attention = self.temporal(
            temporal_input
        )

        temporal_feature = temporal_feature.squeeze(1)

        # -----------------------------
        # Gaze Prediction
        # -----------------------------
        prediction = self.head(
            temporal_feature
        )

        # -----------------------------
        # NEW Spatial Projection
        # -----------------------------
        decoder_input = self.spatial_projection(
            temporal_feature
        )

        # -----------------------------
        # Heatmap Prediction
        # -----------------------------
        heatmap = self.decoder(
            decoder_input
        )

        return prediction, heatmap, attention