```python
from .predictor import Predictor
from dacapo.experiments import Model
from dacapo.experiments.arraytypes import DistanceArray
from dacapo.experiments.datasplits.datasets.arrays import NumpyArray
from dacapo.utils.balance_weights import balance_weights

from funlib.geometry import Coordinate

from scipy.ndimage.morphology import distance_transform_edt
import numpy as np
import torch

import logging
from typing import List

logger = logging.getLogger(__name__)


class InnerDistancePredictor(Predictor):
    """
    This is a class for InnerDistancePredictor.

    Attributes:
        channels (List[str]): The list of strings representing each class being segmented.
        scale_factor (float): A factor to scale distances.

    Methods:
        embedding_dims: Returns the number of classes being segmented.
        create_model: Returns a new model with the given architecture
        create_target: Processes the ground truth data and returns a NumpyArray with distances.
        create_weight: Balances weights independently for each channel.
        output_array_type: Returns a DistanceArray.
        process: Calculates signed distances for a multi-class segmentation task.
        __find_boundaries: Identifies the boundaries within the labels.
        __normalize: Normalizes the distances based on the given norm.
        gt_region_for_roi: Returns the ground truth region for the given region of interest.
        padding: Returns the required padding for the ground truth voxel size.
    """

    def __init__(self, channels: List[str], scale_factor: float):
    """"
    Constructs all the necessary attributes for the InnerDistancePredictor object.
    Params:
        channels (List[str]): list of strings representing each class being segmented.
        scale_factor (float) : a factor to scale distances.
    """
        self.channels = channels
        self.norm = "tanh"
        self.dt_scale_factor = scale_factor

        self.max_distance = 1 * scale_factor
        self.epsilon = 5e-2
        self.threshold = 0.8

    @property
    def embedding_dims(self):
    """ 
    This function returns the count of channels.
    Returns:
        length of the channel list
    """

    def create_model(self, architecture):
    """"
    This function returns a new model with the given architecture.
    Params:
        architecture : architecture of the model
    Returns:
        Model : new model with the given architecture
    """

    def create_target(self, gt):
    """
    This function processes the ground truth data and returns a NumpyArray with distances.
    Params:
        gt : ground truth data
    Returns:
        NumpyArray : array of distances from gt.data
    """

    def create_weight(self, gt, target, mask, moving_class_counts=None):
    """
    This function balances weights independently for each channel.
    Params:
        gt : ground truth data
        target : target data
        mask : mask data
        moving_class_counts : counts of classes in the target
    Returns:
        NumpyArray : weights
        moving_class_counts : counts of classes in the target
    """

    @property
    def output_array_type(self):
    """
    This function returns a DistanceArray.
    Returns:
        DistanceArray :  An array containing distances for a list of items.
    """

    def process(
        self,
        labels: np.ndarray,
        voxel_size: Coordinate,
        normalize=None,
        normalize_args=None,
    ):
    """
    This function calculates signed distances for a multi-class segmentation task.
    Params:
        labels :  labels for the classes
        voxel_size : size of the voxel
        normalize : normalization factor
        normalize_args : arguments for the normalize function
    """

    def __find_boundaries(self, labels):
    """
    This function identifies the boundaries within the labels.
    Params:
        labels :  labels for the classes
    """

    def __normalize(self, distances, norm, normalize_args):
    """
    This function normalizes the distances based on the given norm.
    Params:
        distances : calculated distances
        norm : normalization factor
        normalize_args : arguments for the normalize function
    Returns:
        normalized distances
    """

    def gt_region_for_roi(self, target_spec):
    """
    This function returns the ground truth region for the given region of interest.
    Params:
        target_spec : target specifications
    Returns:
        ground truth region for the region of interest.
    """

    def padding(self, gt_voxel_size: Coordinate) -> Coordinate:
    """
    This function returns the required padding for the ground truth voxel size.
    Params:
        gt_voxel_size : size of the ground truth voxel
    Returns:
        Coordinate : required padding
    """
```