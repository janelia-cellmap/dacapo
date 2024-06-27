from .loss import Loss


class DummyLoss(Loss):
    """
    A class representing a dummy loss function that calculates the absolute difference between each prediction and target.

    Inherits the Loss class.

    Attributes:
        name : str
            name of the loss function
    Methods:
        compute(prediction, target, weight=None)
            Calculate the total loss between prediction and target.
    Note:
        The dummy loss is used to test the training loop and the loss calculation. It is not a real loss function.
        It is used to test the training loop and the loss calculation.

    """

    def compute(self, prediction, target, weight=None):
        """
        Method to calculate the total dummy loss.

        Args:
            prediction : torch.Tensor
                the model's prediction
            target : torch.Tensor
                the target values
            weight : torch.Tensor
                the weight to apply to the loss
        Returns:
            torch.Tensor
                the total loss between prediction and target
        Examples:
            >>> dummy_loss = DummyLoss()
            >>> prediction = torch.tensor([1, 2, 3])
            >>> target = torch.tensor([4, 5, 6])
            >>> dummy_loss.compute(prediction, target)
            tensor(9)
        Note:
            The dummy loss is used to test the training loop and the loss calculation. It is not a real loss function.
            It is used to test the training loop and the loss calculation.
        """

        return abs(prediction - target).sum()
