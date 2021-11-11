from .array import Array

import gunpowder as gp
from funlib.geometry import Coordinate, Roi, roi

import numpy as np


class NumpyArray(Array):
    """This is just a wrapper for a numpy array to make it fit the DaCapo Array interface."""

    def __init__(self, array_config):
        raise RuntimeError("Numpy Array cannot be built from a config file")

    @classmethod
    def from_gp_array(cls, array: gp.Array):
        instance = cls.__new__(cls)
        instance._data = array.data
        instance._dtype = array.data.dtype
        instance._roi = array.spec.roi
        instance._voxel_size = array.spec.voxel_size
        instance._axes = (["c"] if len(array.data.shape) == instance.dims + 1 else []) + [
            "z",
            "y",
            "x",
        ][-instance.dims :]
        return instance

    @classmethod
    def from_np_array(cls, array: np.ndarray, roi, voxel_size, axes):
        instance = cls.__new__(cls)
        instance._data = array
        instance._dtype = array.dtype
        instance._roi = roi
        instance._voxel_size = voxel_size
        instance._axes = axes
        return instance

    @property
    def axes(self):
        return self._axes

    @property
    def dims(self):
        return self._roi.dims

    @property
    def voxel_size(self):
        return self._voxel_size

    @property
    def roi(self):
        return self._roi

    @property
    def writable(self) -> bool:
        return True

    @property
    def data(self):
        return self._data

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def num_channels(self):
        try:
            channel_dim = self.axes.index("c")
            return self.data.shape[channel_dim]
        except ValueError:
            return None

    def __getitem__(self, roi: Roi) -> np.ndarray:
        if not self.roi.contains(roi):
            raise ValueError(f"Cannot fetch data from outside my roi: {self.roi}!")

        assert roi.offset % self.voxel_size == Coordinate(
            (0,) * self.dims
        ), f"Given roi offset: {roi.offset} is not a multiple of voxel_size: {self.voxel_size}"
        assert roi.shape % self.voxel_size == Coordinate(
            (0,) * self.dims
        ), f"Given roi shape: {roi.shape} is not a multiple of voxel_size: {self.voxel_size}"

        offset = (roi.offset - self.roi.offset) / self.voxel_size
        shape = roi.shape / self.voxel_size

        slices = tuple(slice(o, o + s) for o, s in zip(offset, shape))
        if self.axes[0] == "c":
            slices = (slice(None, None),) + slices
        return self.data[slices]
