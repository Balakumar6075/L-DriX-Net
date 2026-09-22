import torch
import torch.nn as nn
from torchvision import models


class EyeStateCNN(nn.Module):

    # Original 2-class mapping
    CLOSED = 0
    OPEN = 1

    NUM_CLASSES = 2

    def __init__(self, pretrained=True):

        super().__init__()

        if pretrained:
            weights = (
                models.MobileNet_V3_Small_Weights.DEFAULT
            )
        else:
            weights = None

        self.model = (
            models.mobilenet_v3_small(
                weights=weights
            )
        )

        input_features = (
            self.model.classifier[-1].in_features
        )

        self.model.classifier[-1] = nn.Linear(
            input_features,
            self.NUM_CLASSES
        )

    def forward(self, x):

        return self.model(x)


def create_model():

    return EyeStateCNN(
        pretrained=True
    )


if __name__ == "__main__":

    model = create_model()

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print("=" * 60)
    print("L-DriX-Net Eye State CNN")
    print("=" * 60)

    print(
        f"Parameters: {total_params:,}"
    )

    print()
    print("Classes:")
    print("0 -> CLOSED")
    print("1 -> OPEN")

    print("=" * 60)