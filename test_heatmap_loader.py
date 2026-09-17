from utils.dataset import LBWDataset

dataset = LBWDataset("dataset/Subject01_1_data")

face, scene, gaze, heatmap = dataset[0]

print("Face     :", face.shape)
print("Scene    :", scene.shape)
print("Gaze     :", gaze.shape)
print("Heatmap  :", heatmap.shape)