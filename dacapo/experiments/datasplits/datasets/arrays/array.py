from funlib.geometry import Coordinate, Roi

import numpy as np

from typing import Optional
from abc import ABC, abstractmethod


class Array(ABC):
    @property
    @abstractmethod
    def axes(self):
        """Returns the axes of this dataset as a string of charactes, as they
        are indexed. Permitted characters are:

            * ``zyx`` for spatial dimensions
            * ``c`` for channels
            * ``s`` for samples
        """
        pass

    @property
    @abstractmethod
    def dims(self) -> int:
        """Returns the number of spatial dimensions."""
        pass

    @property
    @abstractmethod
    def voxel_size(self) -> Coordinate:
        """The size of a voxel in physical units."""
        pass

    @property
    @abstractmethod
    def roi(self) -> Roi:
        """The total ROI of this array, in world units."""
        pass

    @property
    @abstractmethod
    def dtype(self):
        """The dtype of this array, in numpy dtypes"""
        pass

    @property
    @abstractmethod
    def num_channels(self) -> Optional[int]:
        """
        The number of channels provided by this dataset.
        Should return None if the channel dimension doesn't exist.
        """
        pass

    @property
    @abstractmethod
    def data(self):
        """
        Get a numpy like readable and writable view into this array.
        """
        pass

    @property
    @abstractmethod
    def writable(self):
        """
        Can we write to this Array?
        """
        pass

    @abstractmethod
    def __getitem__(self, roi: Roi) -> np.ndarray:
        """
        Get the provided data from this `Array` for the given `roi`.
        """
        pass

    def _can_neuroglance(self):
        return False

    def _neuroglancer_layer(self):
        pass
