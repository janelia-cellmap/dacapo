import torch

from abc import ABC, abstractmethod
from typing import Optional


class Loss(ABC):
    """
    A class used to represent a loss function. This class is an abstract class
    that should be inherited by any loss function class.

    Methods:
        compute(prediction, target, weight) -> torch.Tensor
            Function to compute the loss for the provided prediction and target, with respect to the weight.
    Note:
        This class is abstract. Subclasses must implement the abstract methods. Once created, the values of its attributes
        cannot be changed.
    """

    @abstractmethod
    def compute(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the loss for the given prediction and target. Optionally, if
        given, a loss weight should be considered.

        All arguments are ``torch`` tensors. The return type should be a
        ``torch`` scalar that can be used with an optimizer, just as usual when
        training with ``torch``.

        Args:
            prediction: The predicted tensor.
            target: The target tensor.
            weight: The weight tensor.
        Returns:
            The computed loss tensor.
        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        Examples:
            >>> loss = MSELoss()
            >>> prediction = torch.tensor([1.0, 2.0, 3.0])
            >>> target = torch.tensor([1.0, 2.0, 3.0])
            >>> weight = torch.tensor([1.0, 1.0, 1.0])
            >>> loss.compute(prediction, target, weight)
            tensor(0.)
        Note:
            This method must be implemented in the subclass. It should return the
            computed loss tensor.
        """
        pass
