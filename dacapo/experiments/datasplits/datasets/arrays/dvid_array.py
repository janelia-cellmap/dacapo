from .array import Array
from dacapo.ext import NoSuchModule

try:
    from neuclease.dvid import fetch_info, fetch_labelmap_voxels, fetch_raw
except ImportError:
    fetch_info = NoSuchModule("neuclease.dvid.fetch_info")
    fetch_labelmap_voxels = NoSuchModule("neuclease.dvid.fetch_labelmap_voxels")

from funlib.geometry import Coordinate, Roi
import funlib.persistence

import neuroglancer

import lazy_property
import numpy as np

import logging
from typing import Dict, Tuple, Any, Optional, List

logger = logging.getLogger(__name__)


class DVIDArray(Array):
    """
    This is a DVID array. It is a wrapper around a DVID array that provides
    the necessary methods to interact with the array. It is used to fetch data
    from a DVID server. The source is a tuple of three strings: the server, the UUID,
    and the data name.

    DVID: data management system for terabyte-sized 3D images

    Attributes:
        name (str): The name of the array
        source (tuple[str, str, str]): The source of the array
    Methods:
        __getitem__: Returns the data from the array for a given region of interest
    Notes:
        The source is a tuple of three strings: the server, the UUID, and the data name.
    """

    def __init__(self, array_config):
        """
        Initializes the DVIDArray object

        Args:
            array_config (ArrayConfig): The config object for the array
        Returns:
            DVIDArray: The DVIDArray object
        Raises:
            ValueError: If the array_config is not an ArrayConfig object
        Examples:
            >>> array_config = ArrayConfig(...)
            >>> dvid_array = DVIDArray(array_config)
        Notes:
            The array_config must be an ArrayConfig object.

        """
        super().__init__()
        self.name: str = array_config.name
        self.source: tuple[str, str, str] = array_config.source

    def __str__(self):
        """
        Returns the string representation of the DVIDArray object

        Returns:
            str: The string representation of the DVIDArray object
        Raises:
            ValueError: If the source is not a tuple of three strings
        Examples:
            >>> str(dvid_array)
            DVIDArray(('server', 'UUID', 'data_name'))
        Notes:
            The string representation is the source of the array
        """
        return f"DVIDArray({self.source})"

    def __repr__(self):
        """
        Returns the string representation of the DVIDArray object

        Returns:
            str: The string representation of the DVIDArray object
        Raises:
            ValueError: If the source is not a tuple of three strings
        Examples:
            >>> repr(dvid_array)
            DVIDArray(('server', 'UUID', 'data_name'))
        Notes:  
            The string representation is the source of the array
        """
        return f"DVIDArray({self.source})"

    @lazy_property.LazyProperty
    def attrs(self):
        """
        Returns the attributes of the DVID array

        Returns:
            dict: The attributes of the DVID array
        Raises:
            ValueError: If the attributes is not a dictionary
        Examples:
            >>> dvid_array.attrs
            {'Extended': {'VoxelSize': (1.0, 1.0, 1.0), 'Values': [{'DataType': 'uint64'}]}, 'Extents': {'MinPoint': (0, 0, 0), 'MaxPoint': (100, 100, 100)}}
        Notes:
            The attributes are the same as the source array
        """
        return fetch_info(*self.source)

    @property
    def axes(self):
        """
        Returns the axes of the DVID array

        Returns:
            str: The axes of the DVID array
        Raises:
            ValueError: If the axes is not a string
        Examples:
            >>> dvid_array.axes
            'zyx'
        Notes:
            The axes are the same as the source array
        """
        return ["c", "z", "y", "x"][-self.dims :]

    @property
    def dims(self) -> int:
        """
        Returns the dimensions of the DVID array

        Returns:
            int: The dimensions of the DVID array
        Raises:
            ValueError: If the dimensions is not an integer
        Examples:
            >>> dvid_array.dims
            3
        Notes:
            The dimensions are the same as the source array
        """
        return self.voxel_size.dims

    @lazy_property.LazyProperty
    def _daisy_array(self) -> funlib.persistence.Array:
        """
        Returns the DVID array as a Daisy array

        Returns:
            funlib.persistence.Array: The DVID array as a Daisy array
        Raises:
            ValueError: If the DVID array is not a Daisy array
        Examples:
            >>> dvid_array._daisy_array
            Array(...)
        Notes:
            The DVID array is a Daisy array
        """
        raise NotImplementedError()

    @lazy_property.LazyProperty
    def voxel_size(self) -> Coordinate:
        """
        Returns the voxel size of the DVID array

        Returns:
            Coordinate: The voxel size of the DVID array
        Raises:
            ValueError: If the voxel size is not a Coordinate object
        Examples:
            >>> dvid_array.voxel_size
            Coordinate(x=1.0, y=1.0, z=1.0)
        Notes:
            The voxel size is the same as the source array
        """
        return Coordinate(self.attrs["Extended"]["VoxelSize"])

    @lazy_property.LazyProperty
    def roi(self) -> Roi:
        """
        Returns the region of interest of the DVID array

        Returns:
            Roi: The region of interest of the DVID array
        Raises:
            ValueError: If the region of interest is not a Roi object
        Examples:
            >>> dvid_array.roi
            Roi(...)
        Notes:
            The region of interest is the same as the source array
        """
        return Roi(
            Coordinate(self.attrs["Extents"]["MinPoint"]) * self.voxel_size,
            Coordinate(self.attrs["Extents"]["MaxPoint"]) * self.voxel_size,
        )
        return Roi(
            Coordinate(self.attrs["Extents"]["MinPoint"]) * self.voxel_size,
            Coordinate(self.attrs["Extents"]["MaxPoint"]) * self.voxel_size,
        )

    @property
    def writable(self) -> bool:
        """
        Returns whether the DVID array is writable

        Returns:
            bool: Whether the DVID array is writable
        Raises:
            ValueError: If the writable is not a boolean
        Examples:
            >>> dvid_array.writable
            False
        Notes:
            The writable is the same as the source array
        """
        return False

    @property
    def dtype(self) -> Any:
        """
        Returns the data type of the DVID array

        Returns:
            type: The data type of the DVID array
        Raises:
            ValueError: If the data type is not a type
        Examples:
            >>> dvid_array.dtype
            numpy.uint64
        Notes:  
            The data type is the same as the source array
        """
        return np.dtype(self.attrs["Extended"]["Values"][0]["DataType"])

    @property
    def num_channels(self) -> Optional[int]:
        """
        Returns the number of channels of the DVID array

        Returns:
            int: The number of channels of the DVID array
        Raises:
            ValueError: If the number of channels is not an integer
        Examples:
            >>> dvid_array.num_channels
            1
        Notes:
            The number of channels is the same as the source array
        """
        return None

    @property
    def spatial_axes(self) -> List[str]:
        """
        Returns the spatial axes of the DVID array

        Returns:
            List[str]: The spatial axes of the DVID array
        Raises:
            ValueError: If the spatial axes is not a list
        Examples:
            >>> dvid_array.spatial_axes
            ['z', 'y', 'x']
        Notes:
            The spatial axes are the same as the source array
        """
        return [ax for ax in self.axes if ax not in set(["c", "b"])]

    @property
    def data(self) -> Any:
        """
        Returns the number of channels of the DVID array

        Returns:
            int: The number of channels of the DVID array
        Raises:
            ValueError: If the number of channels is not an integer
        Examples:
            >>> dvid_array.num_channels
            1
        Notes:
            The number of channels is the same as the source array
        """
        raise NotImplementedError()

    def __getitem__(self, roi: Roi) -> np.ndarray[Any, Any]:
        """
        Returns the data of the DVID array for a given region of interest

        Args:
            roi (Roi): The region of interest for which to get the data
        Returns:
            np.ndarray: The data of the DVID array for the region of interest
        Raises:
            ValueError: If the data is not a numpy array
        Examples:
            >>> dvid_array[roi]
            array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]])
        Notes:
            The data is the same as the source array
        """
        box = np.array(
            (roi.offset / self.voxel_size, (roi.offset + roi.shape) / self.voxel_size)
        )
        if self.source[2] == "grayscale":
            data = fetch_raw(*self.source, box)
        elif self.source[2] == "segmentation":
            data = fetch_labelmap_voxels(*self.source, box)
        else:
            raise Exception(self.source)
        return data

    def _can_neuroglance(self) -> bool:
        """
        Returns whether the DVID array can be used with neuroglance

        Returns:
            bool: Whether the DVID array can be used with neuroglance
        Raises:
            ValueError: If the DVID array cannot be used with neuroglance
        Examples:
            >>> dvid_array._can_neuroglance()
            True
        Notes:
            The DVID array can be used with neuroglance
        """
        return True

    def _neuroglancer_source(self):
        """
        Returns the neuroglancer source of the DVID array

        Returns:
            Tuple[str, str, str]: The neuroglancer source of the DVID array
        Raises:
            ValueError: If the neuroglancer source is not a tuple of three strings
        Examples:
            >>> dvid_array._neuroglancer_source()
            ('server', 'UUID', 'data_name')
        Notes:
            The neuroglancer source is the same as the source array
        """
        raise NotImplementedError()

    def _neuroglancer_layer(self) -> Tuple[neuroglancer.ImageLayer, Dict[str, Any]]:
        """
        Returns the neuroglancer layer of the DVID array

        Returns:
            Tuple[neuroglancer.ImageLayer, dict]: The neuroglancer layer of the DVID array
        Raises:
            ValueError: If the neuroglancer layer is not a tuple of an ImageLayer and a dictionary
        Examples:
            >>> dvid_array._neuroglancer_layer()
            (ImageLayer(...), {})
        Notes:
            The neuroglancer layer is the same as the source array
        """
        raise NotImplementedError()

    def _transform_matrix(self):
        """
        Returns the transformation matrix of the DVID array

        Returns:
            np.ndarray: The transformation matrix of the DVID array
        Raises:
            ValueError: If the transformation matrix is not a numpy array
        Examples:
            >>> dvid_array._transform_matrix()
            array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        Notes:
            The transformation matrix is the same as the source array
        """
        raise NotImplementedError()

    def _output_dimensions(self) -> Dict[str, Tuple[float, str]]:
        """
        Returns the output dimensions of the DVID array

        Returns:
            dict: The output dimensions of the DVID array
        Raises:
            ValueError: If the output dimensions is not a dictionary
        Examples:
            >>> dvid_array._output_dimensions()
            {'z': (100, 'nm'), 'y': (100, 'nm'), 'x': (100, 'nm')}
        Notes:
            The output dimensions are the same as the source array
        """
        raise NotImplementedError()

    def _source_name(self) -> str:
        """
        Returns the source name of the DVID array

        Returns:
            str: The source name of the DVID array
        Raises:
            ValueError: If the source name is not a string
        Examples:
            >>> dvid_array._source_name()
            'data_name'
        Notes:
            The source name is the same as the source array
        """
        raise NotImplementedError()

    def add_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Adds metadata to the DVID array

        Args:
            metadata (dict): The metadata to add to the DVID array
        Returns:
            None
        Raises:
            ValueError: If the metadata is not a dictionary
        Examples:
            >>> dvid_array.add_metadata({'description': 'This is a DVID array'})
        Notes:
            The metadata is added to the source array
        """
        raise NotImplementedError()
