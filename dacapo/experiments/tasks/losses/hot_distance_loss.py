from .loss import Loss
import torch


# HotDistance is used for predicting hot and distance maps at the same time.
# The first half of the channels are the hot maps, the second half are the distance maps.
# The loss is the sum of the BCELoss for the hot maps and the MSELoss for the distance maps.
# Model should predict twice the number of channels as the target.
class HotDistanceLoss(Loss):
    

    def compute(self, prediction, target, weight):
        
        target_hot, target_distance = self.split(target)
        prediction_hot, prediction_distance = self.split(prediction)
        weight_hot, weight_distance = self.split(weight)
        return self.hot_loss(
            prediction_hot, target_hot, weight_hot
        ) + self.distance_loss(prediction_distance, target_distance, weight_distance)

    def hot_loss(self, prediction, target, weight):
        
        loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        return torch.mean(loss(prediction, target) * weight)

    def distance_loss(self, prediction, target, weight):
        
        loss = torch.nn.MSELoss()
        return loss(prediction * weight, target * weight)

    def split(self, x):
        
        # Shape[0] is the batch size and Shape[1] is the number of channels.
        assert (
            x.shape[1] % 2 == 0
        ), f"First dimension (Channels) of target {x.shape} must be even to be splitted in hot and distance."
        mid = x.shape[1] // 2
        return torch.split(x, mid, dim=1)
