import gunpowder as gp
import zarr

from pathlib import Path

from .dataset import ArrayDataset


class ZarrDataset(ArrayDataset):
    def __init__(self, filename, ds_name):

        self.filename = filename
        self.ds_name = ds_name

        print(f"Opening zarr container: {filename}, {ds_name}")
        zarr_container = zarr.open(filename)
        ds = zarr_container[ds_name]
        self._voxel_size = gp.Coordinate(ds.attrs["resolution"])
        self._spatial_dims = len(self.voxel_size)
        if "offset" in ds.attrs:
            self._offset = gp.Coordinate(ds.attrs["offset"])
        else:
            self._offset = gp.Coordinate((0,) * self.spatial_dims)
        self._shape = gp.Coordinate(ds.shape)
        self._spatial_shape = gp.Coordinate(self.shape[-self.spatial_dims :])
        self._roi = gp.Roi(self.offset, self.spatial_shape * self.voxel_size)

        if "axes" in ds.attrs:
            self._axes = {d: a for d, a in enumerate(ds.attrs["axes"])}
        else:
            raise ValueError("Dacapo expects an axes attribute for zarr datasets")
            self._axes = {d: d for d in range(len(self.voxel_size))}

        if "c" in self.axes:
            self._num_channels = self.shape[self.axes["c"]]
        else:
            self._num_channels = 0

        if "s" in self.axes:
            self._num_samples = self.shape[self.axes["s"]]
        else:
            self._num_samples = 0

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
    def voxel_size(self):
        return self._voxel_size

    @property
    def spatial_dims(self):
        return self._spatial_dims

    @property
    def offset(self):
        return self._offset

    @property
    def shape(self):
        return self._shape

    @property
    def spatial_shape(self):
        return self._spatial_shape

    @property
    def roi(self):
        return self._roi

    @property
    def axes(self):
        return self._axes

    @property
    def num_channels(self):
        return self._num_channels

    @property
    def num_samples(self):
        return self._num_samples

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
