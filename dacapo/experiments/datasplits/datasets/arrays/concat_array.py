from .array import Array

import gunpowder as gp
from funlib.geometry import Coordinate, Roi, roi

import numpy as np

from typing import Dict, Any


class ConcatArray(Array):
    """This is a wrapper around other `source_array`s that concatenates
    them along the channel dimension."""

    def __init__(self, array_config):
        self.channels = array_config.channels
        self.source_arrays = {
            channel: source_array_config.array_type(source_array_config)
            for channel, source_array_config in array_config.source_array_configs.items()
        }
        self.default_array = (
            array_config.default_config.array_type(array_config.default_config)
            if array_config.default_config is not None
            else None
        )

    @property
    def attrs(self):
        return dict()

    @property
    def source_arrays(self) -> Dict[str, Array]:
        return self._source_arrays

    @source_arrays.setter
    def source_arrays(self, value: Dict[str, Array]):
        assert len(value) > 0, "Source arrays is empty!"
        self._source_arrays = value
        attrs: Dict[str, Any] = {}
        for source_array in value.values():
            axes = attrs.get("axes", source_array.axes)
            assert source_array.axes == axes
            assert axes[0] == "c"
            attrs["axes"] = axes
            roi = attrs.get("roi", source_array.roi)
            assert source_array.roi == roi
            attrs["roi"] = roi
            voxel_size = attrs.get("voxel_size", source_array.voxel_size)
            assert source_array.voxel_size == voxel_size
            attrs["voxel_size"] = voxel_size
        self._source_array = source_array

    @property
    def source_array(self) -> Array:
        return self._source_array

    @property
    def axes(self):
        return self.source_array.axes

    @property
    def dims(self):
        return self.source_array.dims

    @property
    def voxel_size(self):
        return self.source_array.voxel_size

    @property
    def roi(self):
        return self.source_array.roi

    @property
    def writable(self) -> bool:
        return False

    @property
    def data(self):
        raise RuntimeError("Cannot get writable version of this data!")

    @property
    def dtype(self):
        return np.bool

    @property
    def num_channels(self):
        return len(self.channels)

    def __getitem__(self, roi: Roi) -> np.ndarray:
        default = (
            np.zeros_like(self.source_array[roi])
            if self.default_array is None
            else self.default_array[roi]
        )
        arrays = [
            self.source_arrays[channel][roi]
            if channel in self.source_arrays
            else default
            for channel in self.channels
        ]
        shapes = [array.shape for array in arrays]
        ndims = max([len(shape) for shape in shapes])
        assert ndims == len(self.axes), f"{self.axes}, {ndims}"
        shapes = [(1,) * (ndims - len(shape)) + shape for shape in shapes]
        for axis_shapes in zip(*shapes):
            assert max(axis_shapes) == min(axis_shapes), f"{shapes}"
        arrays = [array.reshape(shapes[0]) for array in arrays]
        return np.concatenate(
            arrays,
            axis=0,
        )
