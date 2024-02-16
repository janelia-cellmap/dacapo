"""
This module implements a DistancePredictor class that extends the Predictor
class to include functionality for predicting signed distances for a binary 
segmentation task.

The DistancePredictor class contains various methods to support 
the creation of predictive models, target creation, weight creation and processing.
These predictions are related to the distances deep within background and foreground objects.

"""

class DistancePredictor(Predictor):
    """
    Class for predicting signed distances for a binary segmentation task.

    Attributes:
        channels (list[str]): a list of each class that is being segmented.
        scale_factor (float): affects maximum distance and padding.
        mask_distances (bool): flag for masking distances.
        clipmin (float): the minimum value to clip weight counts to, which by default equals to 0.05.
        clipmax (float): the maximum value to clip weight counts to, which by default equals to 0.95.
    """

    def __init__(
        self,
        channels: List[str],
        scale_factor: float,
        mask_distances: bool,
        clipmin: float = 0.05,
        clipmax: float = 0.95,
    ):
        """
        Initializes a DistancePredictor object.
        """

    ...

    def create_model(self, architecture):
        """
        Creates a 2D or 3D model given an architecture.
        """

    def create_target(self, gt):
        """
        Creates a target from self.process method.        
        """

    ...

    def padding(self, gt_voxel_size: Coordinate) -> Coordinate:
        """
        Calculates the padding needed given gt_voxel_size.

        Args:
            gt_voxel_size (Coordinate): the voxel size from ground truth.

        Returns:
            padding (Coordinate): the padding needed.
        """

    ...

    def __find_boundaries(self, labels):
        """
        Computes boundaries for given labels.
        """

    ...

    def process(self, labels: np.ndarray, voxel_size: Coordinate, normalize=None, normalize_args=None):
        """
        Processes the labels to find their distances.

        Args:
            labels (np.ndarray): array from which distances need to be calculated.
            voxel_size (Coordinate): size of the voxel grid being used.
            normalize : normalization style.
            normalize_args : arguments for normalization method.

        Returns:
            distances (np.ndarray): array having distances.
        """

    ...
