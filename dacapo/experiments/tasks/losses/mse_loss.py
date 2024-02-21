from .loss import Loss
import torch

class MSELoss(Loss):
    """
    A class used to represent the Mean Square Error Loss function (MSELoss).

    Attributes
    ----------
    None

    Methods
    -------
    compute(prediction, target, weight):
        Computes the MSELoss with the given weight for the predictiom amd target.
    """

    def compute(self, prediction, target, weight):
        """
        Function to compute the MSELoss for the provided prediction and target, with respect to the weight.

        Parameters:
        ----------
        prediction : torch.Tensor
            The prediction tensor for which loss needs to be calculated.
        target : torch.Tensor
            The target tensor with respect to which loss is calculated.
        weight : torch.Tensor
            The weight tensor used to weigh the prediction in the loss calculation.

        Returns:
        -------
        torch.Tensor
            The computed MSELoss tensor.
        """
        return torch.nn.MSELoss().forward(prediction * weight, target * weight)