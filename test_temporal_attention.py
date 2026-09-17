import torch

from models.temporal_attention import TemporalAttention

model = TemporalAttention()

# Batch = 8
# Sequence = 5 frames
# Feature = 256

x = torch.randn(8,5,256)

y, attention = model(x)

print("Input Shape      :",x.shape)

print("Output Shape     :",y.shape)

print("Attention Shape  :",attention.shape)