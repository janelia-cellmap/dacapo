from .loss import Loss
import torch


class LinkedCycleLoss(Loss):
    def compute(self, prediction, target, weight):
        pass
        # return torch.nn.MSELoss().forward(prediction * weight, target * weight)
