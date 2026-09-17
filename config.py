import torch
import os


# ===============================
# Dataset
# ===============================

DATASET_PATH = "dataset/Subject01_1_data"


# ===============================
# Training
# ===============================

BATCH_SIZE = 8

NUM_EPOCHS = 10

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-5


# ===============================
# Device
# ===============================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ===============================
# Checkpoints
# ===============================

CHECKPOINT_DIR = "checkpoints"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BEST_MODEL = os.path.join(
    CHECKPOINT_DIR,
    "best_model.pth"
)


# ===============================
# Logging
# ===============================

PRINT_EVERY = 1