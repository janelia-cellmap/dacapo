"""
from .array import Array

import funlib.persistence
from funlib.geometry import Coordinate, Roi

import numpy as np
from skimage.transform import rescale

class ResampledArray(Array):
    """Represents an array that has been resampled.

    Attributes:
        name (str): The name of the array.
        _source_array (Array): The original array before resampling.
        upsample (array-like): The factors by which to upsample along each axis.
        downsample (array-like): The factors by which to downsample along each axis.
        interp_order (int): The interpolation order.
    """

    def __init__(self, array_config):
        """
        Initializes the resampled array with the provided configuration.

        Args:
            array_config (Config): The array configuration.
        """

        ...

    @property
    def attrs(self):
        """Returns the attributes of the source array."""

        ...

    @property
    def axes(self):
        """Returns the axes of the source array."""

        ...

    @property
    def dims(self) -> int:
        """Returns the number of dimensions of the source array."""

        ...

    @property
    def voxel_size(self) -> Coordinate:
        """
        Returns the voxel size in the resampled array. This value is computed as the voxel 
        size in the source array scaled by the downsample factor and divided by the upsample
        factor.
        """

        ...

    @property
    def roi(self) -> Roi:
        """
        Returns the region of interest in the resampled array. 

        This is calculated by snapping the source array's region of interest to 
        the grid defined by the voxel size of the resampled array, using a "shrink" mode.
        """

        ...

    @property
    def writable(self) -> bool:
        """Returns False, as the resampled array is not writable."""

        ...

    @property
    def dtype(self):
        """Returns the data type of the original array."""

        ...

    @property
    def num_channels(self) -> int:
        """Returns the number of channels in the source array."""

        ...

    @property
    def data(self):
        """
        Raises an error if attempting to access directly, as the resampled array is a virtual array.
        """

        ...

    @property
    def scale(self):
        """
        Returns the scaling factors for the spatial dimensions. 

        For each spatial dimension, the scaling factor is computed as the upsample factor divided by 
        the downsample factor.
        """

        ...

    def __getitem__(self, roi: Roi) -> np.ndarray:
        """
        Returns a numpy array with the specified region of interest. 

        Args:
            roi (Roi): The region of interest.
        """

        ...

    def _can_neuroglance(self):
        """Checks if the original array is compatible with Neuroglancer."""

        ...
        
    def _neuroglancer_layer(self):
        """
        Returns the layer configuration for visualizing the array in Neuroglancer.
        """

        ...

    def _source_name(self):
        """Returns the name of the source array."""

        ...
"""
