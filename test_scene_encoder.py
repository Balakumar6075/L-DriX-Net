import torch

from models.scene_encoder import SceneEncoder

model = SceneEncoder()

x = torch.randn(8, 3, 224, 224)

y = model(x)

print("Input :", x.shape)
print("Output:", y.shape)