from .array import Array

from funlib.geometry import Coordinate, Roi

import numpy as np


class DummyArray(Array):
    

    def __init__(self, array_config):
        
        super().__init__()
        self._data = np.zeros((100, 50, 50))

    @property
    def attrs(self):
        
        return dict()

    @property
    def axes(self):
        
        return ["z", "y", "x"]

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
