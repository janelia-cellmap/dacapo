from .loss_abc import LossABC

import torch
import attr

from typing import Optional


@attr.s
class MSELoss(LossABC):

    def instantiate(self):
        return torch.nn.MSELoss()
