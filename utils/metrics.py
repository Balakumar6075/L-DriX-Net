import torch


def accuracy(prediction, target):

    predicted = torch.argmax(prediction, dim=1)

    correct = (predicted == target).sum().item()

    return correct / target.size(0)