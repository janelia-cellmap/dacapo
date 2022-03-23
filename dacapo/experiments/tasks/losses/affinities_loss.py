from .loss import Loss
import torch


class AffinitiesLoss(Loss):
    def __init__(self, num_affinities: int):
        self.num_affinities = num_affinities

    def compute(self, prediction, target, weight):
        affs, affs_target = (
            prediction[:, 0 : self.num_affinities, ...],
            target[:, 0 : self.num_affinities, ...],
        )
        aux, aux_target, aux_weight = (
            prediction[:, self.num_affinities :, ...],
            target[:, self.num_affinities :, ...],
            weight[:, self.num_affinities :, ...],
        )

        return torch.nn.BCEWithLogitsLoss()(affs, affs_target) + torch.nn.MSELoss()(
            torch.nn.Sigmoid()(aux) * aux_weight, aux_target * aux_weight
        )
