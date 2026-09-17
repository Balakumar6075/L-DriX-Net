import torch

from models.decoder import AttentionDecoder

model = AttentionDecoder()

# Simulated feature map from the network
x = torch.randn(8, 256, 14, 14)

heatmap = model(x)

print("Input Feature :", x.shape)
print("Heatmap       :", heatmap.shape)