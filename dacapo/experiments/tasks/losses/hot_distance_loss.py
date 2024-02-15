from .loss import Loss
import torch


class HotDistanceLoss(Loss):
    """
    Loss function used for HotDistance task
    HotDistance is used for predicting hot and distance maps at the same time.
    HotDistanceLoss computes the loss  by summing the BCELoss for the hot maps and the MSELoss for the distance maps.

    Methods:
        compute: Computes the overall loss by combining the hot and distance losses.
        hot_loss: Computes the hot loss between the prediction and target tensors.
        distance_loss: Computes the distance loss between the prediction and target tensors.
        split: Splits the input tensor into hot and distance components.

    """

    def compute(self, prediction, target, weight):
        """
        Computes the loss given the prediction, target, and weight
        by summing the BCELoss for the hot maps and the MSELoss for the distance maps.

        Args:
            prediction (Tensor): The predicted values.
            target (Tensor): The target values.
            weight (Tensor): The weight values.

        Returns:
            Tensor: The computed loss.
        """
        target_hot, target_distance = self._split(target)
        prediction_hot, prediction_distance = self._split(prediction)
        weight_hot, weight_distance = self._split(weight)
        return self._hot_loss(
            prediction_hot, target_hot, weight_hot
        ) + self._distance_loss(prediction_distance, target_distance, weight_distance)

    def _hot_loss(self, prediction, target, weight):
        """
        Computes the hot loss between the prediction and target tensors.

        Args:
            prediction: The predicted hot tensor.
            target: The target hot tensor.
            weight: The weight tensor.

        Returns:
            The hot loss.

        """
        loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        return torch.mean(loss(prediction, target) * weight)

    def _distance_loss(self, prediction, target, weight):
        """
        Computes the distance loss between the prediction and target tensors.

        Args:
            prediction: The predicted distance tensor.
            target: The target distance tensor.
            weight: The weight tensor.

        Returns:
            The distance loss.

        """
        loss = torch.nn.MSELoss()
        return loss(prediction * weight, target * weight)

    def _split(self, x):
        """
        Splits the input tensor into hot and distance components.

        Args:
            x: The input tensor.

        Returns:
            A tuple containing the hot and distance components of the input tensor.

        Raises:
            AssertionError: If the first dimension (channels) of the input tensor is not even.

        """
        assert (
            x.shape[1] % 2 == 0
        ), f"First dimension (Channels) of target {x.shape} must be even to be splitted in hot and distance."
        mid = x.shape[1] // 2
        return torch.split(x, mid, dim=1)
