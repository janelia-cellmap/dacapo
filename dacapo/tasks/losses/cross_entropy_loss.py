from .loss_abc import LossABC

import torch
import attr

from typing import Optional, List


@attr.s
class CrossEntropyLoss(LossABC):
    ignore_label: int = attr.ib(default=-100)
    weights: Optional[List[int]] = attr.ib(default=None)

    def instantiate(self):
        return torch.nn.CrossEntropyLoss(
            torch.tensor(self.weights) if self.weights is not None else None,
            ignore_index=self.ignore_label,
        )
