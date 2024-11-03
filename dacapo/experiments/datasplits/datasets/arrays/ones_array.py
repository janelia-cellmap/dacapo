from .array import Array

from funlib.geometry import Roi

import numpy as np

import logging

logger = logging.getLogger(__name__)


class OnesArray(Array):
    

    def __init__(self, array_config):
        
        logger.warning("OnesArray is deprecated. Use ConstantArray instead.")
        self._source_array = array_config.source_array_config.array_type(
            array_config.source_array_config
        )

    @classmethod
    def like(cls, array: Array):
        
        instance = cls.__new__(cls)
        instance._source_array = array
        return instance

    @property
    def attrs(self):
        
        return dict()

    @property
    def source_array(self) -> Array:
        
        return self._source_array

    @property
    def axes(self):
        
        return self.source_array.axes

    @property
    def dims(self):
        
        return self.source_array.dims

    @property
    def voxel_size(self):
        
        return self.source_array.voxel_size

    @property
    def roi(self):
        
        return self.source_array.roi

    @property
    def writable(self) -> bool:
        
        return False

    @property
    def data(self):
        
        raise RuntimeError("Cannot get writable version of this data!")

    @property
    def dtype(self):
        
        return bool

    @property
    def num_channels(self):
        
        return self.source_array.num_channels

    def __getitem__(self, roi: Roi) -> np.ndarray:
        
        return np.ones_like(self.source_array.__getitem__(roi), dtype=bool)
