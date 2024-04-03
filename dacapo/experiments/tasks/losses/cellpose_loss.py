from .loss import Loss
import torch
from torch import nn

# TODO check support weights


class CellposeLoss(Loss):

    def compute(self, prediction, target, weights=None):
        """loss function between true labels target and prediction prediction"""
        criterion = nn.MSELoss(reduction="mean")
        criterion2 = nn.BCEWithLogitsLoss(reduction="mean")
        veci = 5.0 * target[:, 1:]
        loss = criterion(prediction[:, :-1], veci)
        loss /= 2.0
        loss2 = criterion2(prediction[:, -1], (target[:, 0] > 0.5).float())
        loss = loss + loss2
        return loss
