import torch

from models.icfm import ICFM

model = ICFM()

face = torch.randn(8,256)
scene = torch.randn(8,256)

output = model(face,scene)

print("Face   :", face.shape)
print("Scene  :", scene.shape)
print("----------------------")
print("Output :", output.shape)