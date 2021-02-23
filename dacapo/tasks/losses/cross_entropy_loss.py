from .loss_abc import LossABC

import torch
import attr

from typing import Optional, List


@attr.s
class CrossEntropyLoss(LossABC):
    ignore_label: Optional[int] = attr.ib()
    weights: Optional[List[int]] = attr.ib()

    def loss(self):
        return torch.nn.CrossEntropyLoss(
            torch.tensor(self.weights), ignore_index=self.ignore_label
        )
