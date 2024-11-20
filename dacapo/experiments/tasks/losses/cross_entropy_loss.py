from .loss import Loss
import torch


class CrossEntropyLoss(Loss):
    """
    A closs used for cross entropy tasks. to be used for multi-class classification tasks. each class is represented by a single number.
    recommended for mutually exclusive multi-class classification tasks.
    """

    def __init__(self, weights=None):
        try:
            if weights is not None:
                self.loss_fn = torch.nn.CrossEntropyLoss(
                    weight=torch.tensor(weights, dtype=torch.float32), reduction="mean"
                )
            else:
                self.loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
        except Exception as e:
            raise ValueError(f"Error creating CrossEntropyLoss: {weights} -- {e}")

    def compute(self, prediction, target, weight):
        # class_weights = weight.mean(dim=(0, 2, 3, 4))  # Shape: (num_classes,)

        # # Normalize if needed (optional)
        # class_weights = class_weights / class_weights.sum()
        return self.loss_fn(prediction, target)
