from .array import Array

import gunpowder as gp
from funlib.geometry import Coordinate, Roi

import numpy as np

from typing import List


class NumpyArray(Array):
    """
    This is just a wrapper for a numpy array to make it fit the DaCapo Array interface.

    Attributes:
        data: The numpy array.
        dtype: The data type of the numpy array.
        roi: The region of interest of the numpy array.
        voxel_size: The voxel size of the numpy array.
        axes: The axes of the numpy array.
    Methods:
        from_gp_array: Create a NumpyArray from a Gunpowder Array.
        from_np_array: Create a NumpyArray from a numpy array.
    Note:
        This class is a subclass of Array.
    """

    _data: np.ndarray
    _dtype: np.dtype
    _roi: Roi
    _voxel_size: Coordinate
    _axes: List[str]

    def __init__(self, array_config):
        """
        Create a NumpyArray from an array config.

        Args:
            array_config: The array config.
        Returns:
            NumpyArray: The NumpyArray.
        Raises:
            ValueError: If the array does not have a data type.
        Examples:
            >>> array = NumpyArray(OnesArrayConfig(source_array_config=ArrayConfig()))
            >>> array.data
            array([[[1., 1., 1., 1.],
                    [1., 1., 1., 1.],
                    [1., 1., 1., 1.]],
            <BLANKLINE>
                   [[1., 1., 1., 1.],
                    [1., 1., 1., 1.],
                    [1., 1., 1., 1.]]])
        Note:
            This method creates a NumpyArray from an array config.
        """
        raise RuntimeError("Numpy Array cannot be built from a config file")

    @property
    def attrs(self):
        """
        Returns the attributes of the array.

        Returns:
            dict: The attributes of the array.
        Raises:
            ValueError: If the array does not have attributes.
        Examples:
            >>> array = NumpyArray.from_np_array(np.zeros((2, 3, 4)), Roi((0, 0, 0), (2, 3, 4)), Coordinate((1, 1, 1)), ["z", "y", "x"])
            >>> array.attrs
            {}
        Note:
            This method is a property. It returns the attributes of the array.
        """
        return dict()

    @classmethod
    def from_gp_array(cls, array: gp.Array):
        """
        Create a NumpyArray from a Gunpowder Array.

        Args:
            array (gp.Array): The Gunpowder Array.
        Returns:
            NumpyArray: The NumpyArray.
        Raises:
            ValueError: If the array does not have a data type.
        Examples:
            >>> array = gp.Array(data=np.zeros((2, 3, 4)), spec=gp.ArraySpec(roi=Roi((0, 0, 0), (2, 3, 4)), voxel_size=Coordinate((1, 1, 1))))
            >>> array = NumpyArray.from_gp_array(array)
            >>> array.data
            array([[[0., 0., 0., 0.],
                    [0., 0., 0., 0.],
                    [0., 0., 0., 0.]],
            <BLANKLINE>
                        [[0., 0., 0., 0.],
                        [0., 0., 0., 0.],
                        [0., 0., 0., 0.]]])
        Note:
            This method creates a NumpyArray from a Gunpowder Array.
        """
        instance = cls.__new__(cls)
        instance._data = array.data
        instance._dtype = array.data.dtype
        instance._roi = array.spec.roi
        instance._voxel_size = array.spec.voxel_size
        instance._axes = (
            ((["b", "c"] if len(array.data.shape) == instance.dims + 2 else []))
            + (["c"] if len(array.data.shape) == instance.dims + 1 else [])
            + [
                "c",
                "z",
                "y",
                "x",
            ][-instance.dims :]
        )
        return instance

    @classmethod
    def from_np_array(cls, array: np.ndarray, roi, voxel_size, axes):
        """
        Create a NumpyArray from a numpy array.

        Args:
            array (np.ndarray): The numpy array.
            roi (Roi): The region of interest of the array.
            voxel_size (Coordinate): The voxel size of the array.
            axes (List[str]): The axes of the array.
        Returns:
            NumpyArray: The NumpyArray.
        Raises:
            ValueError: If the array does not have a data type.
        Examples:
            >>> array = NumpyArray.from_np_array(np.zeros((2, 3, 4)), Roi((0, 0, 0), (2, 3, 4)), Coordinate((1, 1, 1)), ["z", "y", "x"])
            >>> array.data
            array([[[0., 0., 0., 0.],
                    [0., 0., 0., 0.],
                    [0., 0., 0., 0.]],
            <BLANKLINE>
                     [[0., 0., 0., 0.],
                      [0., 0., 0., 0.],
                      [0., 0., 0., 0.]]])
        Note:
            This method creates a NumpyArray from a numpy array.

        """
        instance = cls.__new__(cls)
        instance._data = array
        instance._dtype = array.dtype
        instance._roi = roi
        instance._voxel_size = voxel_size
        instance._axes = axes
        return instance

    @property
    def axes(self):
        """
        Returns the axes of the array.

        Returns:
            List[str]: The axes of the array.
        Raises:
            ValueError: If the array does not have axes.
        Examples:
            >>> array = NumpyArray.from_np_array(np.zeros((2, 3, 4)), Roi((0, 0, 0), (2, 3, 4)), Coordinate((1, 1, 1)), ["z", "y", "x"])
            >>> array.axes
            ['z', 'y', 'x']
        Note:
            This method is a property. It returns the axes of the array.
        """
        return self._axes

    @property
    def dims(self):
        """
        Returns the number of dimensions of the array.

        Returns:
            int: The number of dimensions of the array.
        Raises:
            ValueError: If the array does not have a dimension.
        Examples:
            >>> array = NumpyArray.from_np_array(np.zeros((2, 3, 4)), Roi((0, 0, 0), (2, 3, 4)), Coordinate((1, 1, 1)), ["z", "y", "x"])
            >>> array.dims
            3
        Note:
            This method is a property. It returns the number of dimensions of the array.
        """
        return self._roi.dims

    @property
    def voxel_size(self):
        """
        Returns the voxel size of the array.

        Returns:
            Coordinate: The voxel size of the array.
        Examples:
            >>> array = NumpyArray.from_np_array(np.zeros((2, 3, 4)), Roi((0, 0, 0), (2, 3, 4)), Coordinate((1, 1, 1)), ["z", "y", "x"])
            >>> array.voxel_size
            Coordinate((1, 1, 1))
        Note:
            This method is a property. It returns the voxel size of the array.
        """
        return self._voxel_size

    @property
    def roi(self):
        """
        Returns the region of interest of the array.

        Returns:
            Roi: The region of interest of the array.
        Examples:
            >>> array = NumpyArray.from_np_array(np.zeros((2, 3, 4)), Roi((0, 0, 0), (2, 3, 4)), Coordinate((1, 1, 1)), ["z", "y", "x"])
            >>> array.roi
            Roi((0, 0, 0), (2, 3, 4))
        Note:
            This method is a property. It returns the region of interest of the array.
        """
        return self._roi

    @property
    def writable(self) -> bool:
        """
        Returns whether the array is writable.

        Returns:
            bool: Whether the array is writable.
        Raises:
            ValueError: If the array is not writable.
        Examples:
            >>> array = NumpyArray.from_np_array(np.zeros((2, 3, 4)), Roi((0, 0, 0), (2, 3, 4)), Coordinate((1, 1, 1)), ["z", "y", "x"])
            >>> array.writable
            True
        Note:
            This method is a property. It returns whether the array is writable.
        """
        return True

    @property
    def data(self):
        """
        Returns the numpy array.

        Returns:
            np.ndarray: The numpy array.
        Examples:
            >>> array = NumpyArray.from_np_array(np.zeros((2, 3, 4)), Roi((0, 0, 0), (2, 3, 4)), Coordinate((1, 1, 1)), ["z", "y", "x"])
            >>> array.data
            array([[[0., 0., 0., 0.],
                    [0., 0., 0., 0.],
                    [0., 0., 0., 0.]],
            <BLANKLINE>
                   [[0., 0., 0., 0.],
                    [0., 0., 0., 0.],
                    [0., 0., 0., 0.]]])
        Note:
            This method is a property. It returns the numpy array.
        """
        return self._data

    @property
    def dtype(self):
        """
        Returns the data type of the array.

        Returns:
            np.dtype: The data type of the array.
        Raises:
            ValueError: If the array does not have a data type.
        Examples:
            >>> array = NumpyArray.from_np_array(np.zeros((2, 3, 4)), Roi((0, 0, 0), (2, 3, 4)), Coordinate((1, 1, 1)), ["z", "y", "x"])
            >>> array.dtype
            dtype('float64')
        Note:
            This method is a property. It returns the data type of the array.
        """
        return self.data.dtype

    @property
    def num_channels(self):
        """
        Returns the number of channels in the array.

        Returns:
            int: The number of channels in the array.
        Raises:
            ValueError: If the array does not have a channel dimension.
        Examples:
            >>> array = NumpyArray.from_np_array(np.zeros((1, 2, 3, 4)), Roi((0, 0, 0), (1, 2, 3)), Coordinate((1, 1, 1)), ["b", "c", "z", "y", "x"])
            >>> array.num_channels
            1
            >>> array = NumpyArray.from_np_array(np.zeros((2, 3, 4)), Roi((0, 0, 0), (2, 3, 4)), Coordinate((1, 1, 1)), ["z", "y", "x"])
            >>> array.num_channels
            Traceback (most recent call last):
            ...
            ValueError: Array does not have a channel dimension.
        Note:
            This method is a property. It returns the number of channels in the array.
        """
        try:
            channel_dim = self.axes.index("c")
            return self.data.shape[channel_dim]
        except ValueError:
            return None
