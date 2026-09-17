import torch

import config
from models.ldrixnet import LDriXNet


print("=" * 50)
print("Checkpoint Test")
print("=" * 50)

print("Device:", config.DEVICE)
print("Checkpoint:", config.BEST_MODEL)

model = LDriXNet().to(config.DEVICE)

checkpoint = torch.load(
    config.BEST_MODEL,
    map_location=config.DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Checkpoint Epoch :", checkpoint["epoch"])
print("Train Loss       :", checkpoint["train_loss"])
print("Validation Loss  :", checkpoint["val_loss"])
print("Model loaded successfully!")

print("=" * 50)