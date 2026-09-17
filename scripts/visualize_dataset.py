import os
import sys

# Add the project root (L-DriX-Net) to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import matplotlib.pyplot as plt
from utils.dataset import LBWDataset

# Load dataset
dataset = LBWDataset("dataset/Subject01_1_data")

# Get first sample
face, scene = dataset[0]

# Convert tensors to NumPy images
face = face.permute(1, 2, 0).numpy()
scene = scene.permute(1, 2, 0).numpy()

# Display
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(face)
plt.title("Driver Face")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(scene)
plt.title("Road Scene")
plt.axis("off")

plt.tight_layout()
plt.show()