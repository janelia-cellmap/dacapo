from .array import Array

from funlib.geometry import Coordinate, Roi

import lazy_property
import numpy as np
import tifffile

import logging

logger = logging.getLogger(__name__)


class TiffArray(Array):
    """This is a tiff array"""

    def __init__(self, array_config):
        super().__init__()

        self._file_name = array_config.file_name
        self._offset = array_config.offset
        self._voxel_size = array_config.voxel_size
        self._axes = array_config.axes

    @property
    def axes(self):
        return self._axes

    @property
    def dims(self) -> int:
        return self.voxel_size.dims

    @lazy_property.LazyProperty
    def voxel_size(self) -> Coordinate:
        return self._voxel_size

    @lazy_property.LazyProperty
    def roi(self) -> Roi:
        return Roi(self.offset * self.shape)

    @property
    def writable(self):
        return False

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def num_channels(self):
        if "c" in self.axes:
            return self.data.shape[self.axes.index("c")]
        else:
            return None

    @property
    def spatial_axes(self):
        return [c for c in self.axes if c != "c"]

    @lazy_property.LazyProperty
    def data(self):
        return tifffile.TiffFile(self._file_name).values
