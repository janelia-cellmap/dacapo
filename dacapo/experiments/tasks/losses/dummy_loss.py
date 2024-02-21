from .loss import Loss


class DummyLoss(Loss):
    """
    A class representing a dummy loss function that calculates the absolute difference between each prediction and target. 

    Inherits the Loss class.

    Methods
    -------
    compute(prediction, target, weight=None)
        Calculate the total loss between prediction and target.
    """
  

    def compute(self, prediction, target, weight=None):
        """
        Method to calculate the total dummy loss.

        Parameters
        ----------
        prediction : float or int
            predicted output
        target : float or int
            true output
        weight : float or int, optional
            weight parameter for the loss, by default None

        Returns
        -------
        float or int
            Total loss calculated as the sum of absolute differences between prediction and target.
        """
        
        return abs(prediction - target).sum()