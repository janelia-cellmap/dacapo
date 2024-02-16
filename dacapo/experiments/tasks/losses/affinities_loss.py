from .loss import Loss
import torch


class AffinitiesLoss(Loss):
    def __init__(self, num_affinities: int, lsds_to_affs_weight_ratio: float):
        self.num_affinities = num_affinities
        self.lsds_to_affs_weight_ratio = lsds_to_affs_weight_ratio

    def compute(self, prediction, target, weight):
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

        return (
            torch.nn.BCEWithLogitsLoss(reduction="none")(affs, affs_target)
            * affs_weight
        ).mean() + self.lsds_to_affs_weight_ratio * (
            torch.nn.MSELoss(reduction="none")(torch.nn.Sigmoid()(aux), aux_target)
            * aux_weight
        ).mean()
