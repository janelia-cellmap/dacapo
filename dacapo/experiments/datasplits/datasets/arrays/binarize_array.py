from .array import Array

from funlib.geometry import Coordinate, Roi

import neuroglancer

import numpy as np


class BinarizeArray(Array):
    

    def __init__(self, array_config):
        
        self.name = array_config.name
        self._source_array = array_config.source_array_config.array_type(
            array_config.source_array_config
        )
        self.background = array_config.background

        assert (
            "c" not in self._source_array.axes
        ), "Cannot initialize a BinarizeArray with a source array with channels"

        self._groupings = array_config.groupings

    @property
    def attrs(self):
        
        return self._source_array.attrs

    @property
    def axes(self):
        
        return ["c"] + self._source_array.axes

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

    @property
    def channels(self):
        
        return (name for name, _ in self._groupings)

    def __getitem__(self, roi: Roi) -> np.ndarray:
        
        labels = self._source_array[roi]
        grouped = np.zeros((len(self._groupings), *labels.shape), dtype=np.uint8)
        for i, (_, ids) in enumerate(self._groupings):
            if len(ids) == 0:
                grouped[i] += labels != self.background
            for id in ids:
                grouped[i] += labels == id
        return grouped

    def _can_neuroglance(self):
        
        return self._source_array._can_neuroglance()

    def _neuroglancer_source(self):
        
        return self._source_array._neuroglancer_source()

    def _neuroglancer_layer(self):
        
        layer = neuroglancer.SegmentationLayer(source=self._neuroglancer_source())
        return layer

    def _source_name(self):
        
        return self._source_array._source_name()
