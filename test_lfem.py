import torch
from models.lfem import LFEM

model = LFEM()

x = torch.randn(8, 3, 224, 224)

y = model(x)

print("Input :", x.shape)
print("Output:", y.shape)