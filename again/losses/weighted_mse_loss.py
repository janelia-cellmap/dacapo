from torch.nn import MSELoss


class WeightedMSELoss(MSELoss):

    def __init__(self):
        super(WeightedMSELoss, self).__init__()
        self.requires_weights = True

    def forward(self, prediction, target, weights):

        return super(WeightedMSELoss, self).forward(
            prediction*weights,
            target*weights)
