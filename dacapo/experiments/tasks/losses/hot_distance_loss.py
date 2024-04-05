from .loss import Loss
import torch


# HotDistance is used for predicting hot and distance maps at the same time.
# The first half of the channels are the hot maps, the second half are the distance maps.
# The loss is the sum of the BCELoss for the hot maps and the MSELoss for the distance maps.
# Model should predict twice the number of channels as the target.
class HotDistanceLoss(Loss):
    """
    A class used to represent the Hot Distance Loss function. This class inherits from the Loss class. The Hot Distance Loss
    function is used for predicting hot and distance maps at the same time. The first half of the channels are the hot maps,
    the second half are the distance maps. The loss is the sum of the BCELoss for the hot maps and the MSELoss for the distance
    maps. The model should predict twice the number of channels as the target.

    Attributes:
        hot_loss: The Binary Cross Entropy Loss function.
        distance_loss: The Mean Square Error Loss function.
    Methods:
        compute(prediction, target, weight) -> torch.Tensor
            Function to compute the Hot Distance Loss for the provided prediction and target, with respect to the weight.
        split(x) -> Tuple[torch.Tensor, torch.Tensor]
            Function to split the input tensor into two tensors.
    Note:
        This class is abstract. Subclasses must implement the abstract methods. Once created, the values of its attributes
        cannot be changed.

    """

    def compute(self, prediction, target, weight):
        """
        Function to compute the Hot Distance Loss for the provided prediction and target, with respect to the weight.

        Args:
            prediction : torch.Tensor
                The predicted tensor.
            target : torch.Tensor
                The target tensor.
            weight : torch.Tensor
                The weight tensor.
        Returns:
            torch.Tensor
                The computed Hot Distance Loss tensor.
        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        Examples:
            >>> loss = HotDistanceLoss()
            >>> prediction = torch.tensor([1.0, 2.0, 3.0])
            >>> target = torch.tensor([1.0, 2.0, 3.0])
            >>> weight = torch.tensor([1.0, 1.0, 1.0])
            >>> loss.compute(prediction, target, weight)
            tensor(0.)
        Note:
            This method must be implemented in the subclass. It should return the computed Hot Distance Loss tensor.
        """
        target_hot, target_distance = self.split(target)
        prediction_hot, prediction_distance = self.split(prediction)
        weight_hot, weight_distance = self.split(weight)
        return self.hot_loss(
            prediction_hot, target_hot, weight_hot
        ) + self.distance_loss(prediction_distance, target_distance, weight_distance)

    def hot_loss(self, prediction, target, weight):
        """
        The Binary Cross Entropy Loss function. This function computes the BCELoss for the hot maps.

        Args:
            prediction : torch.Tensor
                The predicted tensor.
            target : torch.Tensor
                The target tensor.
            weight : torch.Tensor
                The weight tensor.
        Returns:
            torch.Tensor
                The computed BCELoss tensor.
        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        Examples:
            >>> loss = HotDistanceLoss()
            >>> prediction = torch.tensor([1.0, 2.0, 3.0])
            >>> target = torch.tensor([1.0, 2.0, 3.0])
            >>> weight = torch.tensor([1.0, 1.0, 1.0])
            >>> loss.hot_loss(prediction, target, weight)
            tensor(0.)
        Note:
            This method must be implemented in the subclass. It should return the computed BCELoss tensor.
        """
        loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        return torch.mean(loss(prediction, target) * weight)

    def distance_loss(self, prediction, target, weight):
        """
        The Mean Square Error Loss function. This function computes the MSELoss for the distance maps.

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
            >>> loss = HotDistanceLoss()
            >>> prediction = torch.tensor([1.0, 2.0, 3.0])
            >>> target = torch.tensor([1.0, 2.0, 3.0])
            >>> weight = torch.tensor([1.0, 1.0, 1.0])
            >>> loss.distance_loss(prediction, target, weight)
            tensor(0.)
        Note:
            This method must be implemented in the subclass. It should return the computed MSELoss tensor.
        """
        loss = torch.nn.MSELoss()
        return loss(prediction * weight, target * weight)

    def split(self, x):
        """
        Function to split the input tensor into two tensors.

        Args:
            x : torch.Tensor
                The input tensor.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]
                The two split tensors.
        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        Examples:
            >>> loss = HotDistanceLoss()
            >>> x = torch.tensor([1.0, 2.0, 3.0])
            >>> loss.split(x)
            (tensor([1.0]), tensor([2.0]))
        Note:
            This method must be implemented in the subclass. It should return the two split tensors.
        """
        # Shape[0] is the batch size and Shape[1] is the number of channels.
        assert (
            x.shape[1] % 2 == 0
        ), f"First dimension (Channels) of target {x.shape} must be even to be splitted in hot and distance."
        mid = x.shape[1] // 2
        return torch.split(x, mid, dim=1)
