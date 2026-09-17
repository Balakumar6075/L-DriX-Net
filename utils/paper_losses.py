import torch
import torch.nn as nn
import torch.nn.functional as F


class GazeLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, prediction, target):
        return self.l1(prediction, target)


class KLLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, prediction, target):

        prediction = prediction.view(prediction.size(0), -1)
        target = target.view(target.size(0), -1)

        prediction = F.log_softmax(prediction, dim=1)
        target = F.softmax(target, dim=1)

        return F.kl_div(
            prediction,
            target,
            reduction="batchmean"
        )


class NCCLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, prediction, target):

        prediction = prediction.view(prediction.size(0), -1)
        target = target.view(target.size(0), -1)

        prediction = prediction - prediction.mean(dim=1, keepdim=True)
        target = target - target.mean(dim=1, keepdim=True)

        numerator = (prediction * target).sum(dim=1)

        denominator = torch.sqrt(
            (prediction ** 2).sum(dim=1)
            *
            (target ** 2).sum(dim=1)
            + 1e-8
        )

        ncc = numerator / denominator

        return 1 - ncc.mean()


class TotalLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.gaze_loss = GazeLoss()
        self.kl_loss = KLLoss()
        self.ncc_loss = NCCLoss()

    def forward(
        self,
        pred_gaze,
        gt_gaze,
        pred_heatmap,
        gt_heatmap
    ):

        gaze = self.gaze_loss(
            pred_gaze,
            gt_gaze
        )

        kl = self.kl_loss(
            pred_heatmap,
            gt_heatmap
        )

        ncc = self.ncc_loss(
            pred_heatmap,
            gt_heatmap
        )

        attention = kl + 0.1 * ncc

        total = 2 * gaze + attention

        return total, gaze, kl, ncc