from abc import ABC, abstractmethod


class Loss(ABC):

    @abstractmethod
    def compute(self, prediction, target, weight=None):
        """Compute the loss for the given prediction and target. Optionally, if
        given, a loss weight should be considered.

        All arguments are ``torch`` tensors. The return type should be a
        ``torch`` scalar that can be used with an optimizer, just as usual when
        training with ``torch``."""
        pass
