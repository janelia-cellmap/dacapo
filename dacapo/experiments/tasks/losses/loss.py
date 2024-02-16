Here is the annotated version:

```python
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
        """
        Virtual method to compute the loss for the given prediction and target.

        Args:
            prediction (torch.Tensor): The prediction tensor made by the model. 
            target (torch.Tensor): The actual target tensor against which prediction is to be compared.
            weight (torch.Tensor, optional): The tensor that will be used to apply weightage to the loss. Defaults to None.

        Returns:
            torch.Tensor: The tensor representing computed loss.
        """
        pass

```