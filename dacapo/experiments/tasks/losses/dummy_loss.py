from .loss import Loss


class DummyLoss(Loss):
    def compute(self, prediction, target, weight=None):
        return abs(prediction - target).sum()
