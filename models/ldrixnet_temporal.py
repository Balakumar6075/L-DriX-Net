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


class LDriXNetTemporal(nn.Module):
    """
    Temporal extension of L-DriX-Net.

    Input:
        face  -> [B, T, 3, 224, 224]
        scene -> [B, T, 3, 224, 224]
        gaze  -> [B, T, 24]

    Output:
        prediction         -> [B, 24]
        heatmap            -> [B, 1, 224, 224]
        temporal_attention -> [B, T, T]
        state_logits       -> [B, 3]
    """

    def __init__(
        self,
        temporal_length=16,
        num_states=3
    ):
        super().__init__()

        self.temporal_length = temporal_length

        # ====================================================
        # EXISTING L-DriX-Net COMPONENTS
        # ====================================================

        self.face_encoder = LFEM()

        self.scene_encoder = SceneEncoder()

        self.gaze_encoder = GazeEncoder()

        self.gaze_projection = nn.Linear(
            64,
            256
        )

        self.icfm = ICFM(
            feature_dim=256
        )

        # ====================================================
        # TEMPORAL ATTENTION
        # ====================================================

        self.temporal = TemporalAttention(
            feature_dim=256,
            num_heads=8
        )

        # ====================================================
        # EXISTING GAZE PREDICTION HEAD
        # ====================================================

        self.head = PredictionHead(
            feature_dim=256,
            output_dim=24
        )

        # ====================================================
        # EXISTING HEATMAP BRANCH
        # ====================================================

        self.spatial_projection = SpatialProjection(
            input_dim=256,
            output_channels=256,
            spatial_size=7
        )

        self.decoder = AttentionDecoder()

        # ====================================================
        # NEW DRIVER STATE CLASSIFIER
        #
        # 0 -> CONCENTRATED
        # 1 -> DISTRACTED
        # 2 -> DROWSY
        # ====================================================

        self.state_head = nn.Sequential(
            nn.Linear(
                256,
                128
            ),

            nn.ReLU(),

            nn.Dropout(
                0.2
            ),

            nn.Linear(
                128,
                num_states
            )
        )

    def forward(
        self,
        face,
        scene,
        gaze
    ):

        # ====================================================
        # INPUT VALIDATION
        # ====================================================

        if face.dim() != 5:
            raise ValueError(
                "face must have shape [B,T,C,H,W], "
                f"got {tuple(face.shape)}"
            )

        if scene.dim() != 5:
            raise ValueError(
                "scene must have shape [B,T,C,H,W], "
                f"got {tuple(scene.shape)}"
            )

        if gaze.dim() != 3:
            raise ValueError(
                "gaze must have shape [B,T,24], "
                f"got {tuple(gaze.shape)}"
            )

        batch_size = face.shape[0]
        time_steps = face.shape[1]

        if scene.shape[1] != time_steps:
            raise ValueError(
                "Face and scene sequences must have "
                "the same number of frames."
            )

        if gaze.shape[1] != time_steps:
            raise ValueError(
                "Gaze sequence must have "
                "the same number of frames."
            )

        # ====================================================
        # PROCESS EACH FRAME
        # ====================================================

        temporal_features = []

        for t in range(time_steps):

            # -----------------------------------------------
            # Current frame
            # -----------------------------------------------

            face_t = face[:, t]

            scene_t = scene[:, t]

            gaze_t = gaze[:, t]

            # -----------------------------------------------
            # Face encoding
            # -----------------------------------------------

            face_feature = self.face_encoder(
                face_t
            )

            # -----------------------------------------------
            # Scene encoding
            # -----------------------------------------------

            scene_feature = self.scene_encoder(
                scene_t
            )

            # -----------------------------------------------
            # Gaze encoding
            # -----------------------------------------------

            gaze_feature = self.gaze_encoder(
                gaze_t
            )

            gaze_feature = self.gaze_projection(
                gaze_feature
            )

            # -----------------------------------------------
            # Face + Scene fusion
            # -----------------------------------------------

            fused = self.icfm(
                face_feature,
                scene_feature
            )

            # -----------------------------------------------
            # Add gaze information
            # -----------------------------------------------

            fused = fused + gaze_feature

            temporal_features.append(
                fused
            )

        # ====================================================
        # CREATE TEMPORAL SEQUENCE
        # ====================================================

        # List:
        #
        #   T × [B,256]
        #
        # becomes:
        #
        #   [B,T,256]

        temporal_input = torch.stack(
            temporal_features,
            dim=1
        )

        # ====================================================
        # TEMPORAL SELF-ATTENTION
        # ====================================================

        temporal_output, attention_weights = self.temporal(
            temporal_input
        )

        # ====================================================
        # TEMPORAL POOLING
        # ====================================================

        # MultiheadAttention returns:
        #
        #   [B,T,256]
        #
        # We need one feature vector representing
        # the complete temporal window.

        temporal_feature = temporal_output.mean(
            dim=1
        )

        # ====================================================
        # GAZE PREDICTION
        # ====================================================

        prediction = self.head(
            temporal_feature
        )

        # ====================================================
        # HEATMAP
        # ====================================================

        decoder_input = self.spatial_projection(
            temporal_feature
        )

        heatmap = self.decoder(
            decoder_input
        )

        # ====================================================
        # DRIVER STATE
        # ====================================================

        state_logits = self.state_head(
            temporal_feature
        )

        return (
            prediction,
            heatmap,
            attention_weights,
            state_logits
        )