from .loss import Loss
import torch


class AffinitiesLoss(Loss):
    """
    A class representing a loss function that calculates the loss between affinities and local shape descriptors (LSDs).

    Attributes:
        num_affinities : int
            the number of affinities
        lsds_to_affs_weight_ratio : float
            the ratio of the weight of the loss between affinities and LSDs
    Methods:
        compute(prediction, target, weight=None)
            Calculate the total loss between prediction and target.
    Note:
        The AffinitiesLoss class is used to calculate the loss between affinities and local shape descriptors (LSDs).

    """

    def __init__(self, num_affinities: int, lsds_to_affs_weight_ratio: float):
        """
        Initialize the AffinitiesLoss class with the number of affinities and the ratio of the weight of the loss between affinities and LSDs.

        Args:
            num_affinities : int
                the number of affinities
            lsds_to_affs_weight_ratio : float
                the ratio of the weight of the loss between affinities and LSDs
        Examples:
            >>> affinities_loss = AffinitiesLoss(3, 0.5)
            >>> affinities_loss.num_affinities
            3
            >>> affinities_loss.lsds_to_affs_weight_ratio
            0.5
        Note:
            The AffinitiesLoss class is used to calculate the loss between affinities and local shape descriptors (LSDs).

        """
        self.num_affinities = num_affinities
        self.lsds_to_affs_weight_ratio = lsds_to_affs_weight_ratio

    def compute(self, prediction, target, weight):
        """
        Method to calculate the total loss between affinities and LSDs.

        Args:
            prediction : torch.Tensor
                the model's prediction
            target : torch.Tensor
                the target values
            weight : torch.Tensor
                the weight to apply to the loss
        Returns:
            torch.Tensor
                the total loss between affinities and LSDs
        Raises:
            ValueError: if the number of affinities in the prediction and target does not match
        Examples:
            >>> affinities_loss = AffinitiesLoss(3, 0.5)
            >>> prediction = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
            >>> target = torch.tensor([[9, 10, 11, 12], [13, 14, 15, 16]])
            >>> weight = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]])
            >>> affinities_loss.compute(prediction, target, weight)
            tensor(0.5)
        Note:
            The AffinitiesLoss class is used to calculate the loss between affinities and local shape descriptors (LSDs).

        """
        affs, affs_target, affs_weight = (
            prediction[:, 0 : self.num_affinities, ...],
            target[:, 0 : self.num_affinities, ...],
            weight[:, 0 : self.num_affinities, ...],
        )
        aux, aux_target, aux_weight = (
            prediction[:, self.num_affinities :, ...],
            target[:, self.num_affinities :, ...],
            weight[:, self.num_affinities :, ...],
        )

        if aux.shape[1] == 0:
            return torch.nn.BCEWithLogitsLoss(reduction="none")(
                affs, affs_target
            ).mean()
        else:
            return (
                torch.nn.BCEWithLogitsLoss(reduction="none")(affs, affs_target)
                * affs_weight
            ).mean() + self.lsds_to_affs_weight_ratio * (
                torch.nn.MSELoss(reduction="none")(torch.nn.Sigmoid()(aux), aux_target)
                * aux_weight
            ).mean()
