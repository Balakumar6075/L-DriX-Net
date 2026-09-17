import torch

from utils.metrics import accuracy
from utils.losses import criterion

prediction = torch.randn(8,2)

target = torch.randint(0,2,(8,))

loss = criterion(prediction,target)

acc = accuracy(prediction,target)

print("Loss :",loss.item())

print("Accuracy :",acc)