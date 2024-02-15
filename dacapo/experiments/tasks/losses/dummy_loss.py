from .loss import Loss


class DummyLoss(Loss):
    """A dummy loss function that computes the absolute difference between the prediction and target."""
    
    def compute(self, prediction, target, weight=None):
        return abs(prediction - target).sum()
