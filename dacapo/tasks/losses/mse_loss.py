from .loss_abc import LossABC

import torch
import attr


@attr.s
class MSELoss(LossABC):
    def loss(self):
        return torch.nn.MSELoss()
