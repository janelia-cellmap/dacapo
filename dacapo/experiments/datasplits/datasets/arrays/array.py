from funlib.geometry import Coordinate, Roi

import numpy as np

from typing import Optional, Dict, Any, List, Iterable
from abc import ABC, abstractmethod


class Array(ABC):
    """
    An Array is a multi-dimensional array of data that can be read from and written to. It is
    defined by a region of interest (ROI) in world units, a voxel size, and a number of spatial
    dimensions. The data is stored in a numpy array, and can be accessed using numpy-like slicing
    syntax.

    The Array class is an abstract base class that defines the interface for all Array
    implementations. It provides a number of properties that must be implemented by subclasses,
    such as the ROI, voxel size, and data type of the array. It also provides a method for fetching
    data from the array, which is implemented by slicing the numpy array.

    The Array class also provides a method for checking if the array can be visualized in
    Neuroglancer, and a method for generating a Neuroglancer layer for the array. These methods are
    implemented by subclasses that support visualization in Neuroglancer.

    Attributes:
        attrs (Dict[str, Any]): A dictionary of metadata attributes stored on this array.
        axes (List[str]): The axes of this dataset as a string of characters, as they are indexed.
            Permitted characters are:
                * ``zyx`` for spatial dimensions
                * ``c`` for channels
                * ``s`` for samples
        dims (int): The number of spatial dimensions.
        voxel_size (Coordinate): The size of a voxel in physical units.
        roi (Roi): The total ROI of this array, in world units.
        dtype (Any): The dtype of this array, in numpy dtypes
        num_channels (Optional[int]): The number of channels provided by this dataset. Should return
            None if the channel dimension doesn't exist.
        data (np.ndarray): A numpy-like readable and writable view into this array.
        writable (bool): Can we write to this Array?
    Methods:
        __getitem__(self, roi: Roi) -> np.ndarray: Get a numpy like readable and writable view into
            this array.
        _can_neuroglance(self) -> bool: Check if this array can be visualized in Neuroglancer.
        _neuroglancer_layer(self): Generate a Neuroglancer layer for this array.
        _slices(self, roi: Roi) -> Iterable[slice]: Generate a list of slices for the given ROI.
    Note:
        This class is used to define the interface for all Array implementations. It provides a
        number of properties that must be implemented by subclasses, such as the ROI, voxel size, and
        data type of the array. It also provides a method for fetching data from the array, which is
        implemented by slicing the numpy array. The Array class also provides a method for checking
        if the array can be visualized in Neuroglancer, and a method for generating a Neuroglancer
        layer for the array. These methods are implemented by subclasses that support visualization
        in Neuroglancer.
    """
    @property
    @abstractmethod
    def attrs(self) -> Dict[str, Any]:
        """
        Return a dictionary of metadata attributes stored on this array.

        Returns:
            Dict[str, Any]: A dictionary of metadata attributes stored on this array.
        Raises:
            NotImplementedError: This method must be implemented by the subclass.
        Examples:
            >>> array = Array()
            >>> array.attrs
            {}
        Note:
            This method must be implemented by the subclass.
        """
        pass

    @property
    @abstractmethod
    def axes(self) -> List[str]:
        """
        Returns the axes of this dataset as a string of charactes, as they
        are indexed. Permitted characters are:

            * ``zyx`` for spatial dimensions
            * ``c`` for channels
            * ``s`` for samples
        
        Returns:
            List[str]: The axes of this dataset as a string of characters, as they are indexed.
        Raises:
            NotImplementedError: This method must be implemented by the subclass.
        Examples:
            >>> array = Array()
            >>> array.axes
            ['z', 'y', 'x']
        Note:
            This method must be implemented by the subclass.
        """
        pass

    @property
    @abstractmethod
    def dims(self) -> int:
        """
        Returns the number of spatial dimensions.
        
        Returns:
            int: The number of spatial dimensions.
        Raises:
            NotImplementedError: This method must be implemented by the subclass.
        Examples:
            >>> array = Array()
            >>> array.dims
            3
        Note:
            This method must be implemented by the subclass.
        """
        pass

    @property
    @abstractmethod
    def voxel_size(self) -> Coordinate:
        """
        The size of a voxel in physical units.

        Returns:
            Coordinate: The size of a voxel in physical units.
        Raises:
            NotImplementedError: This method must be implemented by the subclass.
        Examples:
            >>> array = Array()
            >>> array.voxel_size
            Coordinate((1, 1, 1))
        Note:
            This method must be implemented by the subclass.
        """
        pass

    @property
    @abstractmethod
    def roi(self) -> Roi:
        """
        The total ROI of this array, in world units.

        Returns:
            Roi: The total ROI of this array, in world units.
        Raises:
            NotImplementedError: This method must be implemented by the subclass.
        Examples:
            >>> array = Array()
            >>> array.roi
            Roi(offset=Coordinate((0, 0, 0)), shape=Coordinate((100, 100, 100)))
        Note:
            This method must be implemented by the subclass.
        """
        pass

    @property
    @abstractmethod
    def dtype(self) -> Any:
        """
        The dtype of this array, in numpy dtypes
        
        Returns:
            Any: The dtype of this array, in numpy dtypes.
        Raises:
            NotImplementedError: This method must be implemented by the subclass.
        Examples:
            >>> array = Array()
            >>> array.dtype
            np.dtype('uint8')
        Note:
            This method must be implemented by the subclass.
        """
        pass

    @property
    @abstractmethod
    def num_channels(self) -> Optional[int]:
        """
        The number of channels provided by this dataset.
        Should return None if the channel dimension doesn't exist.

        Returns:
            Optional[int]: The number of channels provided by this dataset.
        Raises:
            NotImplementedError: This method must be implemented by the subclass.
        Examples:
            >>> array = Array()
            >>> array.num_channels
            1
        Note:
            This method must be implemented by the subclass.
        """
        pass

    @property
    @abstractmethod
    def data(self) -> np.ndarray:
        """
        Get a numpy like readable and writable view into this array.

        Returns:
            np.ndarray: A numpy like readable and writable view into this array.
        Raises:
            NotImplementedError: This method must be implemented by the subclass.
        Examples:
            >>> array = Array()
            >>> array.data
            np.ndarray
        Note:
            This method must be implemented by the subclass.
        """
        pass

    @property
    @abstractmethod
    def writable(self) -> bool:
        """
        Can we write to this Array?

        Returns:
            bool: Can we write to this Array?
        Raises:
            NotImplementedError: This method must be implemented by the subclass.
        Examples:
            >>> array = Array()
            >>> array.writable
            False
        Note:
            This method must be implemented by the subclass.
        """
        pass

    def __getitem__(self, roi: Roi) -> np.ndarray:
        """
        Get a numpy like readable and writable view into this array.

        Args:
            roi (Roi): The region of interest to fetch data from.
        Returns:
            np.ndarray: A numpy like readable and writable view into this array.
        Raises:
            NotImplementedError: This method must be implemented by the subclass.
        Examples:
            >>> array = Array()
            >>> roi = Roi(offset=Coordinate((0, 0, 0)), shape=Coordinate((100, 100, 100)))
            >>> array[roi]
            np.ndarray
        Note:
            This method must be implemented by the subclass.
        """
        if not self.roi.contains(roi):
            raise ValueError(f"Cannot fetch data from outside my roi: {self.roi}!")

        assert roi.offset % self.voxel_size == Coordinate(
            (0,) * self.dims
        ), f"Given roi offset: {roi.offset} is not a multiple of voxel_size: {self.voxel_size}"
        assert roi.shape % self.voxel_size == Coordinate(
            (0,) * self.dims
        ), f"Given roi shape: {roi.shape} is not a multiple of voxel_size: {self.voxel_size}"

        slices = tuple(self._slices(roi))

        return self.data[slices]

    def _can_neuroglance(self) -> bool:
        """
        Check if this array can be visualized in Neuroglancer.

        Returns:
            bool: Whether this array can be visualized in Neuroglancer.
        Raises:
            NotImplementedError: This method must be implemented by the subclass.
        Examples:
            >>> array = Array()
            >>> array._can_neuroglance()
            False
        Note:
            This method must be implemented by the subclass.
        """
        return False

    def _neuroglancer_layer(self):
        """
        Generate a Neuroglancer layer for this array.

        Raises:
            NotImplementedError: This method must be implemented by the subclass.
        Examples:
            >>> array = Array()
            >>> array._neuroglancer_layer()
            NotImplementedError
        Note:
            This method must be implemented by the subclass.
        """
        pass

    def _slices(self, roi: Roi) -> Iterable[slice]:
        """
        Generate a list of slices for the given ROI.

        Args:
            roi (Roi): The region of interest to generate slices for.
        Returns:
            Iterable[slice]: A list of slices for the given ROI.
        Examples:
            >>> array = Array()
            >>> roi = Roi(offset=Coordinate((0, 0, 0)), shape=Coordinate((100, 100, 100)))
            >>> array._slices(roi)
            [slice(None, None, None), slice(None, None, None), slice(None, None, None)]
        Note:
            This method must be implemented by the subclass.
        """
        offset = (roi.offset - self.roi.offset) / self.voxel_size
        shape = roi.shape / self.voxel_size
        spatial_slices: Dict[str, slice] = {
            a: slice(o, o + s)
            for o, s, a in zip(offset, shape, self.axes[-self.dims :])
        }
        slices: List[slice] = []
        for axis in self.axes:
            if axis == "b" or axis == "c":
                slices.append(slice(None, None))
            else:
                slices.append(spatial_slices[axis])
        return slices
