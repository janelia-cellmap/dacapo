from .predictor import Predictor
from dacapo.experiments import Model
from dacapo.experiments.arraytypes import EmbeddingArray
from dacapo.tmp import np_to_funlib_array

import numpy as np
import torch


class DummyPredictor(Predictor):
    """
    A dummy predictor class that inherits from the base Predictor class.

    Attributes:
        embedding_dims (int): The number of embedding dimensions.
    Methods:
        __init__(self, embedding_dims: int): Initializes a new instance of the DummyPredictor class.
        create_model(self, architecture): Creates a model using the given architecture.
        create_target(self, gt): Creates a target based on the ground truth.
        create_weight(self, gt, target, mask, moving_class_counts=None): Creates a weight based on the ground truth, target, and mask.
        output_array_type: Gets the output array type.
    Notes:
        This is a subclass of Predictor.
    """

    def __init__(self, embedding_dims):
        """
        Initializes a new instance of the DummyPredictor class.

        Args:
            embedding_dims (int): The number of embedding dimensions.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> predictor = DummyPredictor(embedding_dims)
        """
        self.embedding_dims = embedding_dims

    def create_model(self, architecture):
        """
        Creates a model using the given architecture.

        Args:
            architecture: The architecture to use for creating the model.
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

    def create_target(self, gt):
        """
        Creates a target based on the ground truth.

        Args:
            gt: The ground truth.
        Returns:
            NumpyArray: The created target.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> predictor.create_target(gt)
        """
        # zeros
        return np_to_funlib_array(
            np.zeros((self.embedding_dims,) + gt.data.shape[-gt.dims :]),
            gt.roi,
            gt.voxel_size,
            ["c^"] + gt.axis_names,
        )

    def create_weight(self, gt, target, mask, moving_class_counts=None):
        """
        Creates a weight based on the ground truth, target, and mask.

        Args:
            gt: The ground truth.
            target: The target.
            mask: The mask.
            moving_class_counts: The moving class counts.
        Returns:
            Tuple[NumpyArray, None]: The created weight and None.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> predictor.create_weight(gt, target, mask, moving_class_counts)
        """
        # ones
        return (
            np_to_funlib_array(
                np.ones(target.data.shape),
                target.roi,
                target.voxel_size,
                target.axis_names,
            ),
            None,
        )

    @property
    def output_array_type(self):
        """
        Gets the output array type.

        Returns:
            EmbeddingArray: The output array type.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> predictor.output_array_type
        """
        return EmbeddingArray(self.embedding_dims)
