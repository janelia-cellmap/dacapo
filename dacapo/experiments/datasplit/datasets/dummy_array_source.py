from .array_source import ArraySource
from funlib.geometry import Coordinate, Roi
import numpy as np
import zarr


class DummyArraySource(ArraySource):
    """This is just a dummy array source for testing."""

    def __init__(self, source_config):

        self.container = source_config.filename
        self.dataset = 'dummy_data'

        # create a small dummy dataset
        f = zarr.open(self.container, 'a')
        f[self.dataset] = np.zeros((100, 50, 50)).astype(np.float32)
        f[self.dataset].attrs['resolution'] = (1, 2, 2)


    @property
    def axes(self):
        return 'zyx'

    @property
    def dims(self):
        return 3

    @property
    def voxel_size(self):
        return Coordinate(1, 2, 2)

    @property
    def roi(self):
        return Roi((0, 0, 0), (100, 100, 100))
