import torch


class MSELoss(torch.nn.MSELoss):

    def __init__(self):
        super(MSELoss, self).__init__()
        self.requires_weights = False

    def forward(self, prediction, target, weights):
        return super(MSELoss, self).forward(prediction, target)
