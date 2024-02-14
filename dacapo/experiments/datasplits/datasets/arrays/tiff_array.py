from .array import Array

from funlib.geometry import Coordinate, Roi

import lazy_property
import tifffile

import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class TiffArray(Array):
    """This is a tiff array"""

    _offset: Coordinate
    _file_name: Path
    _voxel_size: Coordinate
    _axes: List[str]

    def __init__(self, array_config):
        super().__init__()

        self._file_name = array_config.file_name
        self._offset = array_config.offset
        self._voxel_size = array_config.voxel_size
        self._axes = array_config.axes

    @property
    def attrs(self):
        raise NotImplementedError(
            "Tiffs have tons of different locations for metadata."
        )

    @property
    def axes(self) -> List[str]:
        return self._axes

    @property
    def dims(self) -> int:
        return self.voxel_size.dims

    @lazy_property.LazyProperty
    def shape(self) -> Coordinate:
        data_shape = self.data.shape
        spatial_shape = Coordinate(
            [data_shape[self.axes.index(axis)] for axis in self.spatial_axes]
        )
        return spatial_shape

    @lazy_property.LazyProperty
    def voxel_size(self) -> Coordinate:
        return self._voxel_size

    @lazy_property.LazyProperty
    def roi(self) -> Roi:
        return Roi(self._offset, self.shape)

    @property
    def writable(self) -> bool:
        return False

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def num_channels(self) -> Optional[int]:
        if "c" in self.axes:
            return self.data.shape[self.axes.index("c")]
        else:
            return None

    @property
    def spatial_axes(self) -> List[str]:
        return [c for c in self.axes if c != "c"]

    @lazy_property.LazyProperty
    def data(self):
        return tifffile.TiffFile(self._file_name).values
