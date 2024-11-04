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
        pass
