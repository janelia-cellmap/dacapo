from .array_store import ArrayStore
from funlib.geometry import Coordinate, Roi
import numpy as np
import zarr


class DummyArrayStore(ArrayStore):
    """This is just a dummy array store for testing."""

    def __init__(self, store_config):
        super().__init__()

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
