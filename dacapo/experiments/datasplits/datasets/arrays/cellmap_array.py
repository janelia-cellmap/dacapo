from .array import Array

from funlib.geometry import Coordinate, Roi

import numpy as np


class CellMapArray(Array):
    """This is a zarr array"""

    def __init__(self, array_config):
        super().__init__()
        self._source_array = array_config.source_array_config.array_type(
            array_config.source_array_config
        )

        assert (
            "c" not in self._source_array.axes
        ), f"Cannot initialize a CellMapArray with a source array with channels"

        self._groupings = array_config.groupings

    @property
    def axes(self):
        return "c" + self._source_array.axes

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
        return np.uint8

    @property
    def num_channels(self) -> int:
        return len(self._groupings)

    @property
    def data(self):
        raise ValueError(
            "Cannot get a writable view of this array because it is a virtual "
            "array created by modifying another array on demand."
        )

    def __getitem__(self, roi: Roi) -> np.ndarray:
        labels = self._source_array[roi]
        grouped = np.zeros((len(self._groupings), *labels.shape), dtype=np.uint8)
        for i, ids in enumerate(self._groupings):
            for id in ids:
                grouped[i] += labels == id
        return grouped
