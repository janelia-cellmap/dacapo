from .loss_abc import LossABC

import torch
import attr

from typing import Optional, List


@attr.s
class CrossEntropyLoss(LossABC):
    ignore_label: int = attr.ib(
        default=-100, metadata={"help_text": "A label to avoid training on."}
    )
    weights: Optional[List[int]] = attr.ib(
        default=None,
        metadata={
            "help_text": "A custom weighting to apply to each label. "
            "Weights of [0.5, 0.4, 0.1] would have a higher loss when the target is 0, or 1 vs 2"
        },
    )

    def instantiate(self):
        return torch.nn.CrossEntropyLoss(
            torch.tensor(self.weights) if self.weights is not None else None,
            ignore_index=self.ignore_label,
        )
