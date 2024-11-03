from .array import Array

from funlib.geometry import Coordinate, Roi


import neuroglancer

import numpy as np


class LogicalOrArray(Array):
    

    def __init__(self, array_config):
        
        self.name = array_config.name
        self._source_array = array_config.source_array_config.array_type(
            array_config.source_array_config
        )

    @property
    def axes(self):
        
        return [x for x in self._source_array.axes if x != "c"]

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
    def num_channels(self):
        
        return None

    @property
    def data(self):
        
        raise ValueError(
            "Cannot get a writable view of this array because it is a virtual "
            "array created by modifying another array on demand."
        )

    @property
    def attrs(self):
        
        return self._source_array.attrs

    def __getitem__(self, roi: Roi) -> np.ndarray:
        
        mask = self._source_array[roi]
        if "c" in self._source_array.axes:
            mask = np.max(mask, axis=self._source_array.axes.index("c"))
        return mask

    def _can_neuroglance(self):
        
        return self._source_array._can_neuroglance()

    def _neuroglancer_source(self):
        
        # source_arrays
        if hassattr(self._source_array, "source_arrays"):
            source_arrays = list(self._source_array.source_arrays)
            # apply logical or
            mask = np.logical_or.reduce(source_arrays)
            return mask
        return self._source_array._neuroglancer_source()

    def _combined_neuroglancer_source(self) -> neuroglancer.LocalVolume:
        
        source_array_volume = self._source_array._neuroglancer_source()
        if isinstance(source_array_volume, list):
            source_array_volume = source_array_volume[0]
        result_data = self._neuroglancer_source()

        return neuroglancer.LocalVolume(
            data=result_data,
            dimensions=source_array_volume.dimensions,
            voxel_offset=source_array_volume.voxel_offset,
        )

    def _neuroglancer_layer(self):
        
        # layer = neuroglancer.SegmentationLayer(source=self._neuroglancer_source())
        return neuroglancer.SegmentationLayer(
            source=self._combined_neuroglancer_source()
        )

    def _source_name(self):
        
        name = self._source_array._source_name()
        if isinstance(name, list):
            name = "_".join(name)
        return "logical_or" + name
