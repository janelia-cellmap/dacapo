from .array import Array

from funlib.geometry import Coordinate, Roi

import numpy as np


class IntensitiesArray(Array):
    """
    This is wrapper another array that will normalize intensities to
    the range (0, 1) and convert to float32. Use this if you have your
    intensities stored as uint8 or similar and want your model to
    have floats as input.

    Attributes:
        array_config (ArrayConfig): The config object for the array
        min (float): The minimum intensity value in the array
        max (float): The maximum intensity value in the array
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
        self.name = array_config.name
        self._source_array = array_config.source_array_config.array_type(
            array_config.source_array_config
        )

        self._min = array_config.min
        self._max = array_config.max

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
        Notes:
            The attributes are the same as the source array
        """
        return self._source_array.attrs

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
        return self._source_array.axes

    @property
    def dims(self) -> int:
        """
        Returns the dimensions of the source array

        Returns:
            int: The dimensions of the source array
        Raises:
            ValueError: If the dimensions is not an integer
        Examples:
            >>> intensities_array.dims
            3
        Notes:
            The dimensions are the same as the source array
        """
        return self._source_array.dims

    @property
    def voxel_size(self) -> Coordinate:
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
        return self._source_array.voxel_size

    @property
    def roi(self) -> Roi:
        """
        Returns the region of interest of the source array

        Returns:
            Roi: The region of interest of the source array
        Raises:
            ValueError: If the region of interest is not a Roi object
        Examples:
            >>> intensities_array.roi
            Roi(offset=(0, 0, 0), shape=(10, 20, 30))
        Notes:
            The region of interest is the same as the source array
        """
        return self._source_array.roi

    @property
    def writable(self) -> bool:
        """
        Returns whether the array is writable

        Returns:
            bool: Whether the array is writable
        Raises:
            ValueError: If the array is not writable
        Examples:
            >>> intensities_array.writable
            False
        Notes:
            The array is not writable because it is a virtual array created by modifying another array on demand.
        """
        return False

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
            The data type is always float32
        """
        return np.float32

    @property
    def num_channels(self) -> int:
        """
        Returns the number of channels in the source array

        Returns:
            int: The number of channels in the source array
        Raises:
            ValueError: If the number of channels is not an integer
        Examples:
            >>> intensities_array.num_channels
            3
        Notes:
            The number of channels is the same as the source array
        """
        return self._source_array.num_channels

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
            array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]])
        Notes:
            The data is the same as the source array
        """
        raise ValueError(
            "Cannot get a writable view of this array because it is a virtual "
            "array created by modifying another array on demand."
        )

    def __getitem__(self, roi: Roi) -> np.ndarray:
        """
        Returns the intensities normalized to the range (0, 1)

        Args:
            roi (Roi): The region of interest to get the intensities from
        Returns:
            np.ndarray: The intensities normalized to the range (0, 1)
        Raises:
            ValueError: If the intensities are not in the range (0, 1)
        Examples:
            >>> intensities_array[roi]
            array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]])
        Notes:
            The intensities are normalized to the range (0, 1)
        """
        intensities = self._source_array[roi]
        normalized = (intensities.astype(np.float32) - self._min) / (
            self._max - self._min
        )
        return normalized

    def _can_neuroglance(self):
        """
        Returns whether the array can be visualized with Neuroglancer

        Returns:
            bool: Whether the array can be visualized with Neuroglancer
        Raises:
            ValueError: If the array cannot be visualized with Neuroglancer
        Examples:
            >>> intensities_array._can_neuroglance()
            True
        Notes:
            The array can be visualized with Neuroglancer if the source array can be visualized with Neuroglancer
            
        """
        return self._source_array._can_neuroglance()

    def _neuroglancer_layer(self):
        """
        Returns the Neuroglancer layer of the source array

        Returns:
            dict: The Neuroglancer layer of the source array
        Raises:
            ValueError: If the Neuroglancer layer is not a dictionary
        Examples:
            >>> intensities_array._neuroglancer_layer()
            {'type': 'image', 'source': 'precomputed://https://mybucket.s3.amazonaws.com/mydata'}
        Notes:
            The Neuroglancer layer is the same as the source array
        """
        return self._source_array._neuroglancer_layer()

    def _source_name(self):
        """
        Returns the name of the source array

        Returns:
            str: The name of the source array
        Raises:
            ValueError: If the name is not a string
        Examples:
            >>> intensities_array._source_name()
            'mydata'
        Notes:
            The name is the same as the source array
        """
        return self._source_array._source_name()

    def _neuroglancer_source(self):
        """
        Returns the Neuroglancer source of the source array

        Returns:
            str: The Neuroglancer source of the source array
        Raises:
            ValueError: If the Neuroglancer source is not a string
        Examples:
            >>> intensities_array._neuroglancer_source()
            'precomputed://https://mybucket.s3.amazonaws.com/mydata'
        Notes:
            The Neuroglancer source is the same as the source array
        """
        return self._source_array._neuroglancer_source()
