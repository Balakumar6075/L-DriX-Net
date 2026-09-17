import torch

from models.heads import PredictionHead

model = PredictionHead()

x = torch.randn(8,256)

y = model(x)

print("Input :",x.shape)

print("Output:",y.shape)