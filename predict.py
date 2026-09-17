import os
import torch
import matplotlib.pyplot as plt

import config

from utils.dataset import LBWDataset
from models.ldrixnet import LDriXNet


def predict():

    # -----------------------------
    # Load Dataset
    # -----------------------------
    dataset = LBWDataset(config.DATASET_PATH)

    face, scene, gaze, heatmap = dataset[0]

    face_batch = face.unsqueeze(0).to(config.DEVICE)
    scene_batch = scene.unsqueeze(0).to(config.DEVICE)
    gaze_batch = gaze.unsqueeze(0).to(config.DEVICE)

    # -----------------------------
    # Load Model
    # -----------------------------
    model = LDriXNet().to(config.DEVICE)

    checkpoint = torch.load(
        config.BEST_MODEL,
        map_location=config.DEVICE
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("✅ Model Loaded")

    # -----------------------------
    # Prediction
    # -----------------------------
    with torch.no_grad():

        pred_gaze, pred_heatmap, attention = model(
            face_batch,
            scene_batch,
            gaze_batch
        )

    pred_gaze = pred_gaze.squeeze().cpu()
    pred_heatmap = pred_heatmap.squeeze().cpu()

    # -----------------------------
    # Print Results
    # -----------------------------
    print("\nGround Truth Gaze\n")
    print(gaze)

    print("\nPredicted Gaze\n")
    print(pred_gaze)

    # -----------------------------
    # Create Output Folder
    # -----------------------------
    os.makedirs("outputs", exist_ok=True)

    # -----------------------------
    # Visualization
    # -----------------------------
    plt.figure(figsize=(12,8))

    plt.subplot(2,2,1)
    plt.imshow(face.permute(1,2,0))
    plt.title("Face")
    plt.axis("off")

    plt.subplot(2,2,2)
    plt.imshow(scene.permute(1,2,0))
    plt.title("Scene")
    plt.axis("off")

    plt.subplot(2,2,3)
    plt.imshow(heatmap.squeeze(), cmap="jet")
    plt.title("Ground Truth Heatmap")
    plt.axis("off")

    plt.subplot(2,2,4)
    plt.imshow(pred_heatmap, cmap="jet")
    plt.title("Predicted Heatmap")
    plt.axis("off")

    plt.tight_layout()

    plt.savefig("outputs/prediction_result.png", dpi=300)

    plt.show()

    print("\n✅ Prediction image saved to outputs/prediction_result.png")


if __name__ == "__main__":
    predict()