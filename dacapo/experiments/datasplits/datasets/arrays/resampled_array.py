from .array import Array

import funlib.persistence
from funlib.geometry import Coordinate, Roi

import numpy as np
from skimage.transform import rescale


class ResampledArray(Array):
    """This is a zarr array"""

    def __init__(self, array_config):
        self.name = array_config.name
        self._source_array = array_config.source_array_config.array_type(
            array_config.source_array_config
        )

        self.upsample = Coordinate(max(u, 1) for u in array_config.upsample)
        self.downsample = Coordinate(max(d, 1) for d in array_config.downsample)
        self.interp_order = array_config.interp_order

        assert (
            self.voxel_size * self.upsample
        ) / self.downsample == self._source_array.voxel_size, f"{self.name}, {self._source_array.voxel_size}, {self.voxel_size}, {self.upsample}, {self.downsample}"

    @property
    def attrs(self):
        return self._source_array.attrs

    @property
    def axes(self):
        return self._source_array.axes

    @property
    def dims(self) -> int:
        return self._source_array.dims

    @property
    def voxel_size(self) -> Coordinate:
        return (self._source_array.voxel_size * self.downsample) / self.upsample

    @property
    def roi(self) -> Roi:
        return self._source_array.roi.snap_to_grid(self.voxel_size, mode="shrink")

    @property
    def writable(self) -> bool:
        return False

    @property
    def dtype(self):
        return self._source_array.dtype

    @property
    def num_channels(self) -> int:
        return self._source_array.num_channels

    @property
    def data(self):
        raise ValueError(
            "Cannot get a writable view of this array because it is a virtual "
            "array created by modifying another array on demand."
        )

    @property
    def scale(self):
        spatial_scales = tuple(u / d for d, u in zip(self.downsample, self.upsample))
        if "c" in self.axes:
            scales = list(spatial_scales)
            scales.insert(self.axes.index("c"), 1.0)
            return tuple(scales)
        else:
            return spatial_scales

    def __getitem__(self, roi: Roi) -> np.ndarray:
        snapped_roi = roi.snap_to_grid(self._source_array.voxel_size, mode="grow")
        resampled_array = funlib.persistence.Array(
            rescale(
                self._source_array[snapped_roi].astype(np.float32),
                self.scale,
                order=self.interp_order,
                anti_aliasing=self.interp_order != 0,
            ).astype(self.dtype),
            roi=snapped_roi,
            voxel_size=self.voxel_size,
        )
        return resampled_array.to_ndarray(roi)

    def _neuroglancer_source(self):
        return self._source_array._neuroglancer_source()
    
    def _can_neuroglance(self):
        return self._source_array._can_neuroglance()

    def _neuroglancer_layer(self):
        return self._source_array._neuroglancer_layer()

    def _source_name(self):
        return self._source_array._source_name()
