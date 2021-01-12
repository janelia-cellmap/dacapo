from .loss import Loss
from dacapo.gp import BinarizeNot
import torch


class MaskedMSELoss(torch.nn.MSELoss, Loss):

    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def add_weights(self, target, weights):

        return BinarizeNot(target, weights)

    def forward(self, prediction, target, weights):

        return super(MaskedMSELoss, self).forward(
            prediction*weights,
            target*weights)
