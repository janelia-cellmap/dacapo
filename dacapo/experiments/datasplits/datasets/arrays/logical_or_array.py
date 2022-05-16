from .array import Array

from funlib.geometry import Coordinate, Roi

from fibsem_tools.metadata.groundtruth import Label, LabelList

import neuroglancer

import numpy as np


class LogicalOrArray(Array):
    """ """

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
        return self._source_array._neuroglancer_source()

    def _neuroglancer_layer(self):
        # Generates an Segmentation layer

        layer = neuroglancer.SegmentationLayer(source=self._neuroglancer_source())
        kwargs = {
            "visible": False,
        }
        return layer, kwargs

    def _source_name(self):
        return self._source_array._source_name()
