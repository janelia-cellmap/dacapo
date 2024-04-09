from .array import Array

from funlib.geometry import Coordinate, Roi

import lazy_property
import tifffile

import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class TiffArray(Array):
    """
    This class provides the necessary configuration for a tiff array.

    Attributes:
        _offset: Coordinate
            The offset of the array.
        _file_name: Path
            The file name of the tiff.
        _voxel_size: Coordinate
            The voxel size of the array.
        _axes: List[str]
            The axes of the array.
    Methods:
        attrs() -> Dict
            Return the attributes of the tiff.
    Note:
        This class is a subclass of Array.

    """

    _offset: Coordinate
    _file_name: Path
    _voxel_size: Coordinate
    _axes: List[str]

    def __init__(self, array_config):
        """
        Initialize the TiffArray.

        Args:
            array_config: TiffArrayConfig
                The configuration for the tiff array.
        Raises:
            NotImplementedError:
                Tiffs have tons of different locations for metadata.
        Examples:
            >>> from dacapo.experiments.datasplits.datasets.arrays.tiff_array import TiffArray
            >>> from dacapo.experiments.datasplits.datasets.arrays.tiff_array_config import TiffArrayConfig
            >>> from funlib.geometry import Coordinate
            >>> from pathlib import Path
            >>> tiff_array = TiffArray(TiffArrayConfig(file_name=Path("data.tiff"), offset=Coordinate([0, 0, 0]), voxel_size=Coordinate([1, 1, 1]), axes=["x", "y", "z"]))
        Note:
            This class is a subclass of Array.
        """
        super().__init__()

        self._file_name = array_config.file_name
        self._offset = array_config.offset
        self._voxel_size = array_config.voxel_size
        self._axes = array_config.axes

    @property
    def attrs(self):
        """
        Return the attributes of the tiff.

        Returns:
            Dict: The attributes of the tiff.
        Raises:
            NotImplementedError:
                Tiffs have tons of different locations for metadata.
        Examples:
            >>> tiff_array.attrs
            {'axes': ['x', 'y', 'z'], 'offset': [0, 0, 0], 'voxel_size': [1, 1, 1]}
        Note:
            Tiffs have tons of different locations for metadata.
        """
        raise NotImplementedError(
            "Tiffs have tons of different locations for metadata."
        )

    @property
    def axes(self) -> List[str]:
        """
        Return the axes of the array.

        Returns:
            List[str]: The axes of the array.
        Raises:
            NotImplementedError:
                Tiffs have tons of different locations for metadata.
        Examples:
            >>> tiff_array.axes
            ['x', 'y', 'z']
        Note:
            Tiffs have tons of different locations for metadata.
        """
        return self._axes

    @property
    def dims(self) -> int:
        """
        Return the number of dimensions of the array.

        Returns:
            int: The number of dimensions of the array.
        Raises:
            NotImplementedError:
                Tiffs have tons of different locations for metadata.
        Examples:
            >>> tiff_array.dims
            3
        Note:
            Tiffs have tons of different locations for metadata.
        """
        return self.voxel_size.dims

    @lazy_property.LazyProperty
    def shape(self) -> Coordinate:
        """
        Return the shape of the array.

        Returns:
            Coordinate: The shape of the array.
        Raises:
            NotImplementedError:
                Tiffs have tons of different locations for metadata.
        Examples:
            >>> tiff_array.shape
            Coordinate([100, 100, 100])
        Note:
            Tiffs have tons of different locations for metadata.
        """
        data_shape = self.data.shape
        spatial_shape = Coordinate(
            [data_shape[self.axes.index(axis)] for axis in self.spatial_axes]
        )
        return spatial_shape

    @lazy_property.LazyProperty
    def voxel_size(self) -> Coordinate:
        """
        Return the voxel size of the array.

        Returns:
            Coordinate: The voxel size of the array.
        Raises:
            NotImplementedError:
                Tiffs have tons of different locations for metadata.
        Examples:
            >>> tiff_array.voxel_size
            Coordinate([1, 1, 1])
        Note:
            Tiffs have tons of different locations for metadata.
        """
        return self._voxel_size

    @lazy_property.LazyProperty
    def roi(self) -> Roi:
        """
        Return the region of interest of the array.

        Returns:
            Roi: The region of interest of the array.
        Raises:
            NotImplementedError:
                Tiffs have tons of different locations for metadata.
        Examples:
            >>> tiff_array.roi
            Roi([0, 0, 0], [100, 100, 100])
        Note:
            Tiffs have tons of different locations for metadata.
        """
        return Roi(self._offset, self.shape)

    @property
    def writable(self) -> bool:
        """
        Return whether the array is writable.

        Returns:
            bool: Whether the array is writable.
        Raises:
            NotImplementedError:
                Tiffs have tons of different locations for metadata.
        Examples:
            >>> tiff_array.writable
            False
        Note:
            Tiffs have tons of different locations for metadata.
        """
        return False

    @property
    def dtype(self):
        """
        Return the data type of the array.

        Returns:
            np.dtype: The data type of the array.
        Raises:
            NotImplementedError:
                Tiffs have tons of different locations for metadata.
        Examples:
            >>> tiff_array.dtype
            np.float32
        Note:
            Tiffs have tons of different locations for metadata.

        """
        return self.data.dtype

    @property
    def num_channels(self) -> Optional[int]:
        """
        Return the number of channels of the array.

        Returns:
            Optional[int]: The number of channels of the array.
        Raises:
            NotImplementedError:
                Tiffs have tons of different locations for metadata.
        Examples:
            >>> tiff_array.num_channels
            1
        Note:
            Tiffs have tons of different locations for metadata.

        """
        if "c" in self.axes:
            return self.data.shape[self.axes.index("c")]
        else:
            return None

    @property
    def spatial_axes(self) -> List[str]:
        """
        Return the spatial axes of the array.

        Returns:
            List[str]: The spatial axes of the array.
        Raises:
            NotImplementedError:
                Tiffs have tons of different locations for metadata.
        Examples:
            >>> tiff_array.spatial_axes
            ['x', 'y', 'z']
        Note:
            Tiffs have tons of different locations for metadata.
        """
        return [c for c in self.axes if c != "c"]

    @lazy_property.LazyProperty
    def data(self):
        """
        Return the data of the tiff.

        Returns:
            np.ndarray: The data of the tiff.
        Raises:
            NotImplementedError:
                Tiffs have tons of different locations for metadata.
        Examples:
            >>> tiff_array.data
            np.ndarray
        Note:
            Tiffs have tons of different locations for metadata.
        """
        return tifffile.TiffFile(self._file_name).values
