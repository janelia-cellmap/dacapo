from .array import Array

from funlib.geometry import Coordinate, Roi

import neuroglancer

import numpy as np


class BinarizeArray(Array):
    """
    This is wrapper around a ZarrArray containing uint annotations.
    Because we often want to predict classes that are a combination
    of a set of labels we wrap a ZarrArray with the BinarizeArray
    and provide something like `groupings=[("mito", [3,4,5])]`
    where 4 corresponds to mito_membrane, 5 is mito_ribos, and
    3 is everything else that is part of a mitochondria. The BinarizeArray
    will simply combine labels 3,4,5 into a single binary channel for th
    class of "mito".
    We use a single channel per class because some classes may overlap.
    For example if you had `groupings=[("mito", [3,4,5]), ("membrane", [4, 8, 1])]`
    where 4 is mito_membrane, 8 is er_membrane, and 1 is plasma_membrane.
    Now you can have a binary classification for membrane or not which in
    some cases overlaps with the channel for mitochondria which includes
    the mito membrane.
    """

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
        # Generates an Segmentation layer
        return self._source_array._neuroglancer_layer()

        # layer = neuroglancer.SegmentationLayer(source=self._neuroglancer_source())
        # kwargs = {
        #     "visible": False,
        # }
        # return layer, kwargs

    def _source_name(self):
        return self._source_array._source_name()
