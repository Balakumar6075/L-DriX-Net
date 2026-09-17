import os
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

import config

from utils.dataset import LBWDataset
from models.ldrixnet import LDriXNet
from utils.paper_losses import TotalLoss


def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(8, 5))

    plt.plot(train_losses, label="Train Loss", linewidth=2)
    plt.plot(val_losses, label="Validation Loss", linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("L-DriX-Net Training Curve")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("training_curve.png", dpi=300)
    plt.close()


def train():

    print("=" * 60)
    print("L-DriX-Net Training")
    print("=" * 60)

    # -------------------------------------------------
    # Dataset
    # -------------------------------------------------
    dataset = LBWDataset(config.DATASET_PATH)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # ---------------- TRAIN LOADER ----------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )

    # ---------------- VALIDATION LOADER ----------------
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=True
    )

    print(f"Total Images      : {len(dataset)}")
    print(f"Training Images   : {len(train_dataset)}")
    print(f"Validation Images : {len(val_dataset)}")
    print()

    # -------------------------------------------------
    # Model
    # -------------------------------------------------
    model = LDriXNet().to(config.DEVICE)

    # -------------------------------------------------
    # Optimizer
    # -------------------------------------------------
    optimizer = Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    # -------------------------------------------------
    # Scheduler
    # -------------------------------------------------
    scheduler = StepLR(
        optimizer,
        step_size=25,
        gamma=0.5
    )

    criterion = TotalLoss()

    # -------------------------------------------------
    # LOAD EXISTING BEST MODEL
    # -------------------------------------------------
    start_epoch = 0
    best_val_loss = float("inf")

    if os.path.exists(config.BEST_MODEL):

        print("Loading existing best model...")
        print(f"Checkpoint : {config.BEST_MODEL}")

        checkpoint = torch.load(
            config.BEST_MODEL,
            map_location=config.DEVICE
        )

        # Load model weights
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        # Load optimizer state
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        # Load scheduler state
        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

        # Continue from the next epoch
        start_epoch = checkpoint["epoch"]

        # Restore previous best validation loss
        best_val_loss = checkpoint["val_loss"]

        print(f"Previous Epoch       : {start_epoch}")
        print(f"Previous Train Loss  : {checkpoint['train_loss']:.4f}")
        print(f"Previous Val Loss    : {checkpoint['val_loss']:.4f}")
        print(f"Current Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print()

    else:

        print("No existing checkpoint found.")
        print("Starting training from scratch.")
        print()

    # -------------------------------------------------
    # Training History
    # -------------------------------------------------
    train_losses = []
    val_losses = []

    # -------------------------------------------------
    # Training Loop
    # -------------------------------------------------

    # NUM_EPOCHS means ADDITIONAL epochs
    end_epoch = start_epoch + config.NUM_EPOCHS

    for epoch in range(start_epoch, end_epoch):

        # ===========================
        # TRAIN
        # ===========================
        model.train()

        train_running_loss = 0.0

        for face, scene, gaze, heatmap in train_loader:

            face = face.to(config.DEVICE)
            scene = scene.to(config.DEVICE)
            gaze = gaze.to(config.DEVICE)
            heatmap = heatmap.to(config.DEVICE)

            optimizer.zero_grad()

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

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5
            )

            optimizer.step()

            train_running_loss += total_loss.item()

        scheduler.step()

        train_loss = train_running_loss / len(train_loader)

        # ===========================
        # VALIDATION
        # ===========================
        model.eval()

        val_running_loss = 0.0

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

                val_running_loss += total_loss.item()

        val_loss = val_running_loss / len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch + 1:03d}/{end_epoch:03d}] | "
            f"Train: {train_loss:.4f} | "
            f"Val: {val_loss:.4f} | "
            f"LR: {lr:.6f}"
        )

        # ===========================
        # SAVE BEST MODEL
        # ===========================
        if val_loss < best_val_loss:

            best_val_loss = val_loss

            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            }

            torch.save(
                checkpoint,
                config.BEST_MODEL
            )

            print("✅ Best Validation Model Saved")

    print("=" * 60)
    print("Training Completed")
    print(f"Best Validation Loss : {best_val_loss:.4f}")
    print("=" * 60)

    plot_losses(train_losses, val_losses)

    print("📈 Training curve saved as training_curve.png")


if __name__ == "__main__":
    train()