import torch

from models.fusion import AdaptiveFusion

model = AdaptiveFusion()

face = torch.randn(8, 256)
scene = torch.randn(8, 256)
gaze = torch.randn(8, 64)

output = model(face, scene, gaze)

print("Face  :", face.shape)
print("Scene :", scene.shape)
print("Gaze  :", gaze.shape)
print("-------------------------")
print("Fusion:", output.shape)