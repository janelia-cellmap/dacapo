from .array import Array

from funlib.geometry import Coordinate, Roi
import daisy

import numpy as np
import zarr


class CellMapArray(Array):
    """This is a zarr array"""

    def __init__(self, array_config):
        super().__init__()
        self._source_array = array_config.source_array_config.array_type(
            array_config.source_array_config
        )
        self._groupings = array_config.groupings

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
    def data(self):
        raise ValueError(
            "Cannot get a writable view of this array because it is a virtual "
            "array created by modifying another array on demand."
        )

    def __getitem__(self, roi: Roi) -> np.ndarray:
        # TODO: Fix datatype to be more efficient. bool or uint8
        labels = self._source_array[roi]
        grouped = np.zeros((len(self._groupings), *labels.shape), dtype=labels.dtype)
        for i, ids in enumerate(self._groupings):
            for id in ids:
                grouped[i] += labels == id
        return grouped
