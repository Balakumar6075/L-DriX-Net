from torch.utils.data import DataLoader

import config
from utils.dataset import LBWDataset

dataset = LBWDataset(config.DATASET_PATH)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)

face, scene, gaze, heatmap = next(iter(loader))

print("Face     :", face.shape)
print("Scene    :", scene.shape)
print("Gaze     :", gaze.shape)
print("Heatmap  :", heatmap.shape)