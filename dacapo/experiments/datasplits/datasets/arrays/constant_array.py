from .array import Array

from funlib.geometry import Roi

import numpy as np
import neuroglancer


class ConstantArray(Array):
    def __init__(self, array_config):
        self._source_array = array_config.source_array_config.array_type(
            array_config.source_array_config
        )
        self._constant = array_config.constant

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
        return (
            np.ones_like(self.source_array.__getitem__(roi), dtype=bool)
            * self._constant
        )

    def _can_neuroglance(self):
        return True

    def _neuroglancer_source(self):
        # return self._source_array._neuroglancer_source()
        shape = self.source_array[self.source_array.roi].shape
        return np.ones(shape, dtype=np.uint64) * self._constant

    def _combined_neuroglancer_source(self) -> neuroglancer.LocalVolume:
        source_array_volume = self._source_array._neuroglancer_source()
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
        # return self._source_array._source_name()
        return f"{self._constant}_of_{self.source_array._source_name()}"
