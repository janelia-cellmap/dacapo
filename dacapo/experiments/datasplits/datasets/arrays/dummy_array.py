from .array import Array

from funlib.geometry import Coordinate, Roi

import numpy as np

class DummyArray(Array):
    """This is just a dummy array for testing."""

    def __init__(self, array_config):
        super().__init__()
        self._data = np.zeros(100, 50, 50)

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
