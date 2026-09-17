import torch

from utils.paper_losses import TotalLoss

criterion = TotalLoss()

pred_gaze = torch.randn(8,2)
gt_gaze = torch.randn(8,2)

pred_heatmap = torch.randn(8,1,224,224)
gt_heatmap = torch.randn(8,1,224,224)

loss, gaze, kl, ncc = criterion(
    pred_gaze,
    gt_gaze,
    pred_heatmap,
    gt_heatmap
)

print("Total Loss :", loss.item())
print("Gaze Loss  :", gaze.item())
print("KL Loss    :", kl.item())
print("NCC Loss   :", ncc.item())