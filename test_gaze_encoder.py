import torch

from models.gaze_encoder import GazeEncoder

model = GazeEncoder()

x = torch.randn(8,24)

y = model(x)

print("Input :",x.shape)

print("Output:",y.shape)