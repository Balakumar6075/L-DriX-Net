from torch.utils.data import DataLoader
from utils.dataset import LBWDataset

dataset = LBWDataset("dataset/Subject01_1_data")

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)

face, scene, gaze = next(iter(loader))

print("Face Batch :", face.shape)
print("Scene Batch:", scene.shape)
print("Gaze Batch :", gaze.shape)