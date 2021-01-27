from .loss import Loss
import gunpowder as gp
import torch

from .gp import AddDistance


class WeightedMSELoss(torch.nn.MSELoss, Loss):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def add_weights(self, target, weights):

        return gp.BalanceLabels(labels=target, scales=weights)

    def forward(self, prediction, target, weights):

        return super(WeightedMSELoss, self).forward(
            prediction * weights, target * weights
        )


class DistanceWeightedMSELoss(torch.nn.MSELoss, Loss):
    def __init__(self):
        super(DistanceWeightedMSELoss, self).__init__()

    def add_weights(self, target, weights):

        return AddDistance(labels=target, scales=weights)

    def forward(self, prediction, target, weights):

        return super(DistanceWeightedMSELoss, self).forward(
            prediction * weights, target * weights
        )
