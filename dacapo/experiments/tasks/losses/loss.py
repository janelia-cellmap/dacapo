import torch

from abc import ABC, abstractmethod
from typing import Optional


class Loss(ABC):
    @abstractmethod
    def compute(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the loss for the given prediction and target. Optionally, if
        given, a loss weight should be considered.

        All arguments are ``torch`` tensors. The return type should be a
        ``torch`` scalar that can be used with an optimizer, just as usual when
        training with ``torch``."""
        pass
