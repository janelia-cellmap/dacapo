from .loss_abc import LossABC

import torch
import attr

from typing import Optional


@attr.s
class WeightedMSELossConfig(LossABC):

    def instantiate(self):
        return WeightedMSELoss()


class WeightedMSELoss(torch.nn.MSELoss):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, prediction, target, weights):

        return super(WeightedMSELoss, self).forward(
            prediction * weights, target * weights
        )
