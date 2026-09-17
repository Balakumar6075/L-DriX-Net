from utils.dataset import LBWDataset

dataset = LBWDataset("dataset/Subject01_1_data")

face, scene, gaze = dataset[0]

print(face.shape)
print(scene.shape)

print("\nAvailable Gaze Keys")

for key in gaze:
    print(key)