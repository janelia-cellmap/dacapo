"""
from dacapo.experiments.arraytypes.probabilities import ProbabilityArray
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


class HotDistancePredictor(Predictor):
    """
    This class is primarily used to predict hot distances for binary segmentation tasks. It can also predict multiple classes for segmentation.

    Attributes:
        channels (List[str]): The list of classes to be segmented.
        scale_factor (float): The scale factor for distance transformation.
        mask_distances (bool): Indicator to mask the distance or not.
        
    """

    def __init__(self, channels: List[str], scale_factor: float, mask_distances: bool):
        """
        Args:
            channels (List[str]): The list of classes to be segmented.
            scale_factor (float): The scale factor for distance transformation.
            mask_distances (bool): Indicator to mask the distance or not.
        """
        # your code

    # your methods

    def create_model(self, architecture):
        """
        Creates a model for the given architecture.

        Args:
            architecture (Architecture): The deep learning architecture to be used.

        Returns:
            Model: The model that was created.
        """
        # your code

    def create_target(self, gt):
        """
        Creates the target for training from the given ground truth data.

        Args:
            gt (np.array): Ground truth data.

        Returns:
            NumpyArray: Processed target data.
        """
        # your code

    def create_weight(self, gt, target, mask, moving_class_counts=None):
        """
        Computes the weight for each channel independently.

        Args:
            gt (np.array): Ground truth data.
            target (NumpyArray): The desired target output.
            mask (np.array): Masking array to be applied.
            moving_class_counts (int, optional): Class counts that are moving. Defaults to None.

        Returns:
            tuple: A tuple containing the weight and class counts.
        """
        # your code

    @property
    def output_array_type(self):
        """
        Output array type information (TODO: Needs more description)

        Returns:
            ProbabilityArray: A Probability array object.
        """
        # your code

    def create_distance_mask(self, distances: np.ndarray, mask: np.ndarray, voxel_size: Coordinate, normalize=None, normalize_args=None):
        """
        Creates a distance mask.

        Args:
            distances (np.ndarray): An array with distances information.
            mask (np.ndarray): A binary mask to apply.
            voxel_size (Coordinate): The voxel size to use.
            normalize (str, optional): The normalization to apply. Defaults to None.
            normalize_args (dict, optional): Arguments for the normalization method. Defaults to None.

        Returns:
            np.ndarray: The created distance mask.
        """
        # your code

    def process(self, labels: np.ndarray, voxel_size: Coordinate, normalize=None, normalize_args=None):
        """
        Runs the main process for the given label and voxel size.

        Args:
            labels (np.ndarray): An array with label information.
            voxel_size (Coordinate): The voxel size to use.
            normalize (str, optional): The normalization to apply. Defaults to None.
            normalize_args (dict, optional): Arguments for the normalization method. Defaults to None.

        Returns:
            np.ndarray: Processed label data.
        """
        # your code

    # Private methods are still explained for the purpose of developers

    def gt_region_for_roi(self, target_spec):
        """
        Computes the ground truth region for a given region of interest.

        Args:
            target_spec (NumpyArray): A region of interest.

        Returns:
            NumpyArray: The ground truth region.
        """
        # your code

    def padding(self, gt_voxel_size: Coordinate) -> Coordinate:
        """
        Computes the padding for the given ground truth voxel size.

        Args:
            gt_voxel_size (Coordinate): The voxel size of the ground truth.

        Returns:
            Coordinate: The computed padding.
        """
        # your code
"""