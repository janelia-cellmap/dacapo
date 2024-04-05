from .array import Array

from funlib.geometry import Coordinate, Roi

import numpy as np


class DummyArray(Array):
    """
    This is just a dummy array for testing. It has a shape of (100, 50, 50) and is filled with zeros.

    Attributes:
        array_config (ArrayConfig): The config object for the array
    Methods:
        __getitem__: Returns the intensities normalized to the range (0, 1)
    Notes:
        The array_config must be an ArrayConfig object.
        The min and max values are used to normalize the intensities.
        All intensities are converted to float32.
    
    """

    def __init__(self, array_config):
        """
        Initializes the IntensitiesArray object

        Args:
            array_config (ArrayConfig): The config object for the array
        Raises:
            ValueError: If the array_config is not an ArrayConfig object
        Examples:
            >>> array_config = ArrayConfig(...)
            >>> intensities_array = IntensitiesArray(array_config)
        Notes:
            The array_config must be an ArrayConfig object.
        """
        super().__init__()
        self._data = np.zeros((100, 50, 50))

    @property
    def attrs(self):
        """
        Returns the attributes of the source array

        Returns:
            dict: The attributes of the source array
        Raises:
            ValueError: If the attributes is not a dictionary
        Examples:
            >>> intensities_array.attrs
            {'resolution': (1.0, 1.0, 1.0), 'unit': 'micrometer'}
        """
        return dict()

    @property
    def axes(self):
        """
        Returns the axes of the source array

        Returns:
            str: The axes of the source array
        Raises:
            ValueError: If the axes is not a string
        Examples:
            >>> intensities_array.axes
            'zyx'
        Notes:
            The axes are the same as the source array
        """
        return ["z", "y", "x"]

    @property
    def dims(self):
        """
        Returns the number of dimensions of the source array    

        Returns:
            int: The number of dimensions of the source array
        Raises:
            ValueError: If the dims is not an integer
        Examples:
            >>> intensities_array.dims
            3
        Notes:
            The dims are the same as the source array
        """
        return 3

    @property
    def voxel_size(self):
        """
        Returns the voxel size of the source array

        Returns:
            Coordinate: The voxel size of the source array
        Raises:
            ValueError: If the voxel size is not a Coordinate object
        Examples:
            >>> intensities_array.voxel_size
            Coordinate(x=1.0, y=1.0, z=1.0)
        Notes:
            The voxel size is the same as the source array
        """
        return Coordinate(1, 2, 2)

    @property
    def roi(self):
        """
        Returns the region of interest of the source array

        Returns:
            Roi: The region of interest of the source array
        Raises:
            ValueError: If the roi is not a Roi object
        Examples:
            >>> intensities_array.roi
            Roi(offset=(0, 0, 0), shape=(100, 100, 100))
        Notes:
            The roi is the same as the source array
        """
        return Roi((0, 0, 0), (100, 100, 100))

    @property
    def writable(self) -> bool:
        """
        Returns whether the array is writable

        Returns:
            bool: Whether the array is writable
        Examples:
            >>> intensities_array.writable
            True
        Notes:
            The array is always writable
        """
        return True

    @property
    def data(self):
        """
        Returns the data of the source array

        Returns:
            np.ndarray: The data of the source array
        Raises:
            ValueError: If the data is not a numpy array
        Examples:
            >>> intensities_array.data
            array([[[0., 0., 0., ..., 0., 0., 0.],
                    [0., 0., 0., ..., 0., 0., 0.],
                    [0., 0., 0., ..., 0., 0., 0.],
                    ...,
                    [0., 0., 0., ..., 0., 0., 0.],
                    [0., 0., 0., ..., 0., 0., 0.],
                    [0., 0., 0., ..., 0., 0., 0.]],
        Notes:
            The data is the same as the source array
        """
        return self._data

    @property
    def dtype(self):
        """
        Returns the data type of the array

        Returns:
            type: The data type of the array
        Raises:
            ValueError: If the data type is not a type
        Examples:
            >>> intensities_array.dtype
            numpy.float32
        Notes:
            The data type is the same as the source array
        """
        return self._data.dtype

    @property
    def num_channels(self):
        """
        Returns the number of channels in the source array

        Returns:
            int: The number of channels in the source array
        Raises:
            ValueError: If the number of channels is not an integer
        Examples:
            >>> intensities_array.num_channels
            1
        Notes:
            The number of channels is the same as the source array
        """
        return None
