import torch

from models.ldrixnet import LDriXNet

model = LDriXNet()

face = torch.randn(2,3,224,224)
scene = torch.randn(2,3,224,224)
gaze = torch.randn(2,24)

prediction, heatmap, attention = model(face, scene, gaze)

print("Prediction :", prediction.shape)
print("Heatmap    :", heatmap.shape)
print("Attention  :", attention.shape)