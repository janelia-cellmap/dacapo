from .array import Array

from funlib.geometry import Coordinate, Roi

import numpy as np


class DummyArray(Array):
    """This is just a dummy array for testing."""

    def __init__(self, array_config):
        super().__init__()
        self._data = np.zeros((100, 50, 50))

    @property
    def axes(self):
        return "zyx"

    @property
    def dims(self):
        return 3

    @property
    def voxel_size(self):
        return Coordinate(1, 2, 2)

    @property
    def roi(self):
        return Roi((0, 0, 0), (100, 100, 100))

    @property
    def writable(self) -> bool:
        return True

    @property
    def data(self):
        return self._data

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def num_channels(self):
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

        offset = roi.offset / self.voxel_size
        shape = roi.shape / self.voxel_size

        return self.data[tuple(slice(o, o+s) for o, s in zip(offset, shape))]
