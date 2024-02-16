from funlib.geometry import Coordinate, Roi

import numpy as np

from typing import Optional, Dict, Any, List, Iterable
from abc import ABC, abstractmethod


class Array(ABC):
    """
    Abstract class representing an n-dimensional array with some associated meta-data such as 
    number of channels, dimensions, voxel size etc. and utilities to manipulate and view the data.
    """
    @property
    @abstractmethod
    def attrs(self) -> Dict[str, Any]:
        """
        Abstract method to return dictionary of meta-data attributes.

        Returns: 
            Dict[str, Any]: Dictionary containing meta-data attributes.
        """
        pass

    @property
    @abstractmethod
    def axes(self) -> List[str]:
        """
        Abstract method to return axes.

        Returns:
            List[str]: List of axes.
        """
        pass

    @property
    @abstractmethod
    def dims(self) -> int:
        """
        Abstract method to return number of dimensions.

        Returns:
            int: Number of dimensions.
        """
        pass

    @property
    @abstractmethod
    def voxel_size(self) -> Coordinate:
        """
        Abstract method to return voxel size.

        Returns:
            Coordinate: Size of voxel.
        """
        pass

    @property
    @abstractmethod
    def roi(self) -> Roi:
        """
        Abstract method to return roi (region of interest).

        Returns:
            Roi: Region of interest.
        """
        pass

    @property
    @abstractmethod
    def dtype(self) -> Any:
        """
        Abstract method to return data type of the array.

        Returns:
            Any: Data type of the array.
        """
        pass

    @property
    @abstractmethod
    def num_channels(self) -> Optional[int]:
        """
        Abstract method to return number of channels.

        Returns:
            Optional[int]: Number of channels if present else None.
        """
        pass

    @property
    @abstractmethod
    def data(self) -> np.ndarray:
        """
        Abstract method to return a numpy ndarray view of the data.

        Returns:
            np.ndarray: Numpy ndarray view of the data.
        """
        pass

    @property
    @abstractmethod
    def writable(self) -> bool:
        """
        Abstract method to check if data is writable.

        Returns:
            bool: True if data is writable, False otherwise.
        """
        pass

    def __getitem__(self, roi: Roi) -> np.ndarray:
        """
        Method to return a subset of the data defined by a region of interest.

        Args:
            roi (Roi): The region of interest.

        Returns:
            np.ndarray: Data within the provided region of interest.

        Raises:
            ValueError: If the provided region of interest is outside the total ROI of the array.
            AssertionError: If the offset of ROI is not multiple of voxel size.
            AssertionError: If the shape of ROI is not multiple of voxel size.
        """
        pass  # implementation details omitted in this abstract class for brevity    

    def _can_neuroglance(self) -> bool:
        """
        Method to check if data can be visualized using neuroglance.

        Returns:
            bool: Always returns False.
        """
        pass  # implementation details omitted in this docstring for brevity

    def _neuroglancer_layer(self):
        """
        Method to generate neuroglancer layer.

        Note: The functionality is not implemented in this method.
        """
        pass  # implementation details omitted in this docstring for brevity

    def _slices(self, roi: Roi) -> Iterable[slice]:
        """
        Method to generate slices for a given region of interest.

        Args:
            roi (Roi): The region of interest.

        Returns:
            Iterable[slice]: Iterable of slices generated from provided roi.
        """
        pass  # implementation details omitted in this docstring for brevity