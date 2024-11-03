from .array import Array

from funlib.geometry import Coordinate, Roi

import numpy as np


class IntensitiesArray(Array):
    def __init__(self, array_config):
        self.name = array_config.name
        self._source_array = array_config.source_array_config.array_type(
            array_config.source_array_config
        )

        self._min = array_config.min
        self._max = array_config.max

    @property
    def attrs(self):
        return self._source_array.attrs

    @property
    def axes(self):
        return self._source_array.axes

    @property
    def dims(self) -> int:
        return self._source_array.dims

    @property
    def voxel_size(self) -> Coordinate:
        return self._source_array.voxel_size

    @property
    def roi(self) -> Roi:
        return self._source_array.roi

    @property
    def writable(self) -> bool:
        return False

    @property
    def dtype(self):
        return np.float32

    @property
    def num_channels(self) -> int:
        return self._source_array.num_channels

    @property
    def data(self):
        raise ValueError(
            "Cannot get a writable view of this array because it is a virtual "
            "array created by modifying another array on demand."
        )

    def __getitem__(self, roi: Roi) -> np.ndarray:
        intensities = self._source_array[roi]
        normalized = (intensities.astype(np.float32) - self._min) / (
            self._max - self._min
        )
        return normalized

    def _can_neuroglance(self):
        return self._source_array._can_neuroglance()

    def _neuroglancer_layer(self):
        return self._source_array._neuroglancer_layer()

    def _source_name(self):
        return self._source_array._source_name()

    def _neuroglancer_source(self):
        return self._source_array._neuroglancer_source()
