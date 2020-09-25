import gunpowder as gp
import zarr

import logging

from .dataset import ArrayDataset

logger = logging.getLogger(__name__)


class DacapoConventionError(Exception):
    pass


class ZarrDataset(ArrayDataset):
    def __init__(self, filename, ds_name):

        self.filename = filename
        self.ds_name = ds_name

        zarr_container = zarr.open(filename)
        ds = zarr_container[ds_name]

        # necessary fields:
        self._shape = gp.Coordinate(ds.shape)
        if "axes" in ds.attrs:
            self._axes = {d: a for d, a in enumerate(ds.attrs["axes"])}
            self._inv_axes = {a: d for d, a in self._axes.items()}
            self._channel_dim = self._inv_axes.get("c")
            self._sample_dim = self._inv_axes.get("s")
            self._spatial_axes = sorted(
                [
                    i
                    for i in self._axes.keys()
                    if i != self.channel_dim and i != self.sample_dim
                ]
            )
        else:
            raise DacapoConventionError(
                "Dacapo expects zarr arrays to come with axis labels. "
                "Note that label 'c' is reserved for the (optional) channel dimension. "
                "Label 's' is reserved for the (optional) sample dimension. "
                "Any other label will be treated as a spatial dimension."
            )

        # optional fields
        if "resolution" in ds.attrs:
            self._voxel_size = gp.Coordinate(ds.attrs["resolution"])
        else:
            self._voxel_size = gp.Coordinate(tuple(1 for i in self._spatial_axes))
        if "offset" in ds.attrs:
            self._offset = gp.Coordinate(ds.attrs["offset"])
        else:
            self._offset = gp.Coordinate(tuple(0 for i in self._spatial_axes))

        # gt specific
        if "num_classes" in ds.attrs:
            self._num_classes = ds.attrs["num_classes"]
        else:
            self._num_classes = 0
        if "background_label" in ds.attrs:
            self._background_label = ds.attrs["background_label"]
        else:
            self._background_label = None

    @property
    def shape(self):
        return self._shape

    @property
    def axes(self):
        return self._axes

    @property
    def channel_dim(self):
        return self._channel_dim

    @property
    def sample_dim(self):
        return self._sample_dim

    @property
    def spatial_axes(self):
        return self._spatial_axes

    @property
    def voxel_size(self):
        return self._voxel_size

    @property
    def spatial_dims(self):
        return len(self.spatial_axes)

    @property
    def offset(self):
        return self._offset

    @property
    def spatial_shape(self):
        return gp.Coordinate(tuple(self.shape[i] for i in self._spatial_axes))

    @property
    def roi(self):
        return gp.Roi(self.offset, self.spatial_shape * self.voxel_size)

    @property
    def num_channels(self):
        if "c" in self.axes:
            return self.shape[self.channel_dim]
        else:
            return 0

    @property
    def num_samples(self):
        if "s" in self.axes:
            return self.shape[self.sample_dim]
        else:
            return 0

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def background_label(self):
        return self._background_label

    def get_source(self, array, overwrite_spec=None):

        if overwrite_spec:
            return gp.ZarrSource(
                self.filename,
                {array: self.ds_name},
                array_specs={array: overwrite_spec},
            )
        else:
            return gp.ZarrSource(self.filename, {array: self.ds_name})
