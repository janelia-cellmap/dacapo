from .array import Array
from funlib.geometry import Coordinate, Roi
import numpy as np


class DummyArray(Array):
    """
    A dummy array class for testing. Inherits from the Array class.
    
    Attributes:
        _data (numpy array): A zeros numpy array of shape (100, 50, 50).
    
    Methods:
        attrs: Returns a dictionary.
        axes: Returns an array of axes.
        dims: Returns the dimensions of the array.
        voxel_size: Returns the size of the voxel.
        roi: Returns the region of interest.
        writable: Returns true.
        data: Returns the data of the array.
        dtype: Returns the data type of the array.
        num_channels: Returns None.
    """

    def __init__(self, array_config):
        """
        Constructs the DummyArray object.
        
        Args:
            array_config: The configuration settings for the array.
        """
        super().__init__()
        self._data = np.zeros((100, 50, 50))

    @property
    def attrs(self):
        """Returns a dictionary."""
        return dict()

    @property
    def axes(self):
        """Returns a list of axes ['z', 'y', 'x']."""
        return ["z", "y", "x"]

    @property
    def dims(self):
        """Returns the dimensions of the array, in this case, 3."""
        return 3

    @property
    def voxel_size(self):
        """
        Returns the size of the voxel as a Coordinate object with values (1, 2, 2).
        """
        return Coordinate(1, 2, 2)

    @property
    def roi(self):
        """
        Returns the region of interest as a Roi object with values ((0,0,0), (100,100,100)).
        """
        return Roi((0, 0, 0), (100, 100, 100))

    @property
    def writable(self) -> bool:
        """Always returns True."""
        return True

    @property
    def data(self):
        """Returns the _data attribute with zeros numpy array."""
        return self._data

    @property
    def dtype(self):
        """Returns the data type of the _data attribute."""
        return self._data.dtype

    @property
    def num_channels(self):
        """Currently hardcoded to return None."""
        return None
