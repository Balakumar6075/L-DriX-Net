import torch
import torch.nn as nn


class TemporalAttention(nn.Module):

    def __init__(self,
                 feature_dim=256,
                 num_heads=8):

        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, x):

        # x : (Batch, Sequence, Features)

        attended, weights = self.attention(
            x,
            x,
            x
        )

        x = self.norm(attended + x)

        return x, weights