import os
import math
import torch

from torch.utils.data import DataLoader, random_split

import config

from utils.dataset import LBWDataset
from models.ldrixnet import LDriXNet
from utils.paper_losses import TotalLoss


def validate():

    print("=" * 60)
    print("L-DriX-Net Validation")
    print("=" * 60)

    # -----------------------------
    # Dataset
    # -----------------------------
    dataset = LBWDataset(config.DATASET_PATH)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    _, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )

    print(f"Validation Images : {len(val_dataset)}")
    print()

    # -----------------------------
    # Model
    # -----------------------------
    model = LDriXNet().to(config.DEVICE)

    checkpoint = torch.load(
        config.BEST_MODEL,
        map_location=config.DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("✅ Loaded:", config.BEST_MODEL)
    print()

    criterion = TotalLoss()

    total_loss_sum = 0.0
    gaze_loss_sum = 0.0
    kl_loss_sum = 0.0
    ncc_loss_sum = 0.0

    mae_sum = 0.0
    mse_sum = 0.0
    total_samples = 0

    os.makedirs("outputs", exist_ok=True)

    with torch.no_grad():

        for face, scene, gaze, heatmap in val_loader:

            face = face.to(config.DEVICE)
            scene = scene.to(config.DEVICE)
            gaze = gaze.to(config.DEVICE)
            heatmap = heatmap.to(config.DEVICE)

            pred_gaze, pred_heatmap, attention = model(
                face,
                scene,
                gaze
            )

            total_loss, gaze_loss, kl_loss, ncc_loss = criterion(
                pred_gaze,
                gaze,
                pred_heatmap,
                heatmap
            )

            total_loss_sum += total_loss.item()
            gaze_loss_sum += gaze_loss.item()
            kl_loss_sum += kl_loss.item()
            ncc_loss_sum += ncc_loss.item()

            mae = torch.mean(torch.abs(pred_gaze - gaze))
            mse = torch.mean((pred_gaze - gaze) ** 2)

            batch_size = face.size(0)

            mae_sum += mae.item() * batch_size
            mse_sum += mse.item() * batch_size

            total_samples += batch_size

    avg_total = total_loss_sum / len(val_loader)
    avg_gaze = gaze_loss_sum / len(val_loader)
    avg_kl = kl_loss_sum / len(val_loader)
    avg_ncc = ncc_loss_sum / len(val_loader)

    mae = mae_sum / total_samples
    rmse = math.sqrt(mse_sum / total_samples)

    print("=" * 60)
    print("Validation Results")
    print("=" * 60)
    print(f"Total Loss : {avg_total:.4f}")
    print(f"Gaze Loss  : {avg_gaze:.4f}")
    print(f"KL Loss    : {avg_kl:.4f}")
    print(f"NCC Loss   : {avg_ncc:.4f}")
    print(f"MAE        : {mae:.4f}")
    print(f"RMSE       : {rmse:.4f}")
    print("=" * 60)

    with open("outputs/metrics.txt", "w") as f:

        f.write("L-DriX-Net Validation Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"Total Loss : {avg_total:.4f}\n")
        f.write(f"Gaze Loss  : {avg_gaze:.4f}\n")
        f.write(f"KL Loss    : {avg_kl:.4f}\n")
        f.write(f"NCC Loss   : {avg_ncc:.4f}\n")
        f.write(f"MAE        : {mae:.4f}\n")
        f.write(f"RMSE       : {rmse:.4f}\n")

    print("✅ Metrics saved to outputs/metrics.txt")


if __name__ == "__main__":
    validate()