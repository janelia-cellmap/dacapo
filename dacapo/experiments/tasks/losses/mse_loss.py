from .loss import Loss
import torch


class MSELoss(Loss):
    """
    A class used to represent the Mean Square Error Loss function (MSELoss). This class inherits from the Loss class.

    Methods:
        compute(prediction, target, weight) -> torch.Tensor
            Function to compute the MSELoss for the provided prediction and target, with respect to the weight.
    Note:
        This class is abstract. Subclasses must implement the abstract methods. Once created, the values of its attributes
        cannot be changed.
    """

    def compute(self, prediction, target, weight):
        """
        Function to compute the MSELoss for the provided prediction and target, with respect to the weight.

        Args:
            prediction : torch.Tensor
                The predicted tensor.
            target : torch.Tensor
                The target tensor.
            weight : torch.Tensor
                The weight tensor.
        Returns:
            torch.Tensor
                The computed MSELoss tensor.
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
            This method must be implemented in the subclass. It should return the computed MSELoss tensor.
        """
        return torch.nn.MSELoss().forward(prediction * weight, target * weight)
