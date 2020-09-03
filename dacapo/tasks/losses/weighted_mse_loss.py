from .loss import Loss
import gunpowder as gp
import torch


class WeightedMSELoss(torch.nn.MSELoss, Loss):

    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def add_weights(self, target, weights):

        return gp.BalanceLabels(
            labels=target,
            scales=weights)

    def forward(self, prediction, target, weights):

        return super(WeightedMSELoss, self).forward(
            prediction*weights,
            target*weights)
