import torch

from models.ldrixnet_temporal import LDriXNetTemporal


CHECKPOINT = "checkpoints/best_model.pth"


def main():

    print("=" * 60)
    print("L-DriX-Net Temporal Model Test")
    print("=" * 60)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # --------------------------------------------------------
    # Create temporal model
    # --------------------------------------------------------

    model = LDriXNetTemporal(
        temporal_length=16,
        num_states=3
    )

    model = model.to(device)

    # --------------------------------------------------------
    # Load existing L-DriX-Net checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device
    )

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    else:

        state_dict = checkpoint

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False
    )

    print()
    print("Missing keys:")
    
    for key in missing:
        print("  ", key)

    print()
    print("Unexpected keys:")

    for key in unexpected:
        print("  ", key)

    # --------------------------------------------------------
    # Create dummy 16-frame input
    # --------------------------------------------------------

    TEMPORAL_LENGTH = 16

    face = torch.randn(
        1,
        TEMPORAL_LENGTH,
        3,
        224,
        224,
        device=device
    )

    scene = torch.randn(
        1,
        TEMPORAL_LENGTH,
        3,
        224,
        224,
        device=device
    )

    gaze = torch.randn(
        1,
        TEMPORAL_LENGTH,
        24,
        device=device
    )

    print()
    print("INPUT SHAPES")
    print("-" * 60)

    print(
        "Face:",
        face.shape
    )

    print(
        "Scene:",
        scene.shape
    )

    print(
        "Gaze:",
        gaze.shape
    )

    # --------------------------------------------------------
    # Evaluation mode
    # --------------------------------------------------------

    model.eval()

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    with torch.no_grad():

        (
            prediction,
            heatmap,
            attention,
            state_logits
        ) = model(
            face,
            scene,
            gaze
        )

    # --------------------------------------------------------
    # Output shapes
    # --------------------------------------------------------

    print()
    print("OUTPUT SHAPES")
    print("-" * 60)

    print(
        "Prediction:",
        prediction.shape
    )

    print(
        "Heatmap:",
        heatmap.shape
    )

    print(
        "Temporal attention:",
        attention.shape
    )

    print(
        "State logits:",
        state_logits.shape
    )

    # --------------------------------------------------------
    # State probabilities
    # --------------------------------------------------------

    state_probabilities = torch.softmax(
        state_logits,
        dim=1
    )

    predicted_state = torch.argmax(
        state_probabilities,
        dim=1
    ).item()

    state_names = [
        "CONCENTRATED",
        "DISTRACTED",
        "DROWSY"
    ]

    print()
    print("STATE OUTPUT")
    print("-" * 60)

    for i, name in enumerate(state_names):

        probability = (
            state_probabilities[0, i]
            .item()
        )

        print(
            f"{name}: "
            f"{probability:.4f}"
        )

    print()
    print(
        "Predicted state:",
        state_names[predicted_state]
    )

    print()
    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()