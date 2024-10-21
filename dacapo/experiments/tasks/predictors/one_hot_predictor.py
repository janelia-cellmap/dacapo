from .predictor import Predictor
from dacapo.experiments import Model
from dacapo.experiments.arraytypes import ProbabilityArray
from dacapo.tmp import np_to_funlib_array
from funlib.persistence import Array

import numpy as np
import torch

from typing import List
import logging

logger = logging.getLogger(__name__)


class OneHotPredictor(Predictor):
    """
    A predictor that uses one-hot encoding for classification tasks.

    Attributes:
        classes (List[str]): The list of class labels.
    Methods:
        __init__(self, classes: List[str]): Initializes the OneHotPredictor.
        create_model(self, architecture): Create the model for the predictor.
        create_target(self, gt): Create the target array for training.
        create_weight(self, gt, target, mask, moving_class_counts=None): Create the weight array for training.
        output_array_type: Get the output array type.
        process(self, labels: np.ndarray): Process the labels array and convert it to one-hot encoding.
    Notes:
        This is a subclass of Predictor.
    """

    def __init__(self, classes: List[str]):
        """
        Initialize the OneHotPredictor.

        Args:
            classes (List[str]): The list of class labels.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> predictor = OneHotPredictor(classes)
        """
        self.classes = classes

    @property
    def embedding_dims(self):
        """
        Get the number of embedding dimensions.

        Returns:
            int: The number of embedding dimensions.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> embedding_dims = predictor.embedding_dims
        """
        return len(self.classes)

    def create_model(self, architecture):
        """
        Create the model for the predictor.

        Args:
            architecture: The architecture for the model.
        Returns:
            Model: The created model.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> model = predictor.create_model(architecture)
        """
        head = torch.nn.Conv3d(
            architecture.num_out_channels, self.embedding_dims, kernel_size=3
        )

        return Model(architecture, head)

    def create_target(self, gt: Array):
        """
        Turn labels into a one hot encoding
        """
        label_data = gt[:]
        if gt.channel_dims == 0:
            label_data = label_data[np.newaxis]
        elif gt.channel_dims > 1:
            raise ValueError(f"Cannot handle multiple channel dims: {gt.channel_dims}")
        one_hots = self.process(label_data)
        return np_to_funlib_array(
            one_hots,
            gt.roi.offset,
            gt.voxel_size,
        )

    def create_weight(self, gt, target, mask, moving_class_counts=None):
        """
        Create the weight array for training.

        Args:
            gt: The ground truth array.
            target: The target array.
            mask: The mask array.
            moving_class_counts: The moving class counts.
        Returns:
            Tuple[NumpyArray, None]: The created weight array and None.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> predictor.create_weight(gt, target, mask, moving_class_counts)

        """
        return (
            np_to_funlib_array(
                np.ones(target.data.shape),
                target.roi.offset,
                target.voxel_size,
            ),
            None,
        )

    @property
    def output_array_type(self):
        """
        Get the output array type.

        Returns:
            ProbabilityArray: The output array type.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> output_array_type = predictor.output_array_type
        """
        return ProbabilityArray(self.classes)

    def process(
        self,
        labels: np.ndarray,
    ):
        """
        Process the labels array and convert it to one-hot encoding.

        Args:
            labels (np.ndarray): The labels array.
        Returns:
            np.ndarray: The one-hot encoded array.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> one_hots = predictor.process(labels)
        Notes:
            Assumes labels has a singleton channel dim and channel dim is first.
        """
        # TODO: Assumes labels has a singleton channel dim and channel dim is first
        one_hots = np.zeros((self.embedding_dims,) + labels.shape[1:], dtype=np.uint8)
        for i, _ in enumerate(self.classes):
            one_hots[i] += labels[0] == i
        return one_hots
