from .array import Array

import gunpowder as gp
from funlib.geometry import Coordinate, Roi

import numpy as np

from typing import List


class NumpyArray(Array):
    

    _data: np.ndarray
    _dtype: np.dtype
    _roi: Roi
    _voxel_size: Coordinate
    _axes: List[str]

    def __init__(self, array_config):
        
        raise RuntimeError("Numpy Array cannot be built from a config file")

    @property
    def attrs(self):
        
        return dict()

    @classmethod
    def from_gp_array(cls, array: gp.Array):
        
        instance = cls.__new__(cls)
        instance._data = array.data
        instance._dtype = array.data.dtype
        instance._roi = array.spec.roi
        instance._voxel_size = array.spec.voxel_size
        instance._axes = (
            ((["b", "c"] if len(array.data.shape) == instance.dims + 2 else []))
            + (["c"] if len(array.data.shape) == instance.dims + 1 else [])
            + [
                "c",
                "z",
                "y",
                "x",
            ][-instance.dims :]
        )
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
