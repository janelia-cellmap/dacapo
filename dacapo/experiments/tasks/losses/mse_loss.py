from .loss import Loss
import torch


class MSELoss(Loss):
    def compute(self, prediction, target, weight):
        return torch.nn.MSELoss().forward(prediction * weight, target * weight)
