import gunpowder as gp
import zarr


class Dataset:

    def __init__(self, filename, ds_name):

        self.filename = filename
        self.ds_name = ds_name

        ds = zarr.open(filename)[ds_name]
        self.voxel_size = gp.Coordinate(ds.attrs['resolution'])
        self.spatial_dims = len(self.voxel_size)
        if 'offset' in ds.attrs:
            self.offset = gp.Coordinate(ds.attrs['offset'])
        else:
            self.offset = gp.Coordinate((0,)*self.spatial_dims)
        self.shape = gp.Coordinate(ds.shape)
        self.spatial_shape = gp.Coordinate(self.shape[-self.spatial_dims:])
        self.roi = gp.Roi(self.offset, self.spatial_shape*self.voxel_size)

        if 'axes' in ds.attrs:
            self.axes = {
                d: a
                for d, a in enumerate(ds.attrs['axes'])
            }
        else:
            self.axes = {d: d for d in range(len(self.voxel_size))}

        if 'c' in self.axes:
            self.num_channels = self.shape[self.axes['c']]
        else:
            self.num_channels = 0

        if 's' in self.axes:
            self.num_samples = self.shape[self.axes['s']]
        else:
            self.num_samples = 0

        # gt specific

        if 'num_classes' in ds.attrs:
            self.num_classes = ds.attrs['num_classes']
        else:
            self.num_classes = 0
        if 'background_label' in ds.attrs:
            self.background_label = ds.attrs['background_label']
        else:
            self.background_label = None

    def get_source(self, array, overwrite_spec=None):

        if overwrite_spec:
            return gp.ZarrSource(
                self.filename,
                {array: self.ds_name},
                array_specs={array: overwrite_spec})
        else:
            return gp.ZarrSource(
                self.filename,
                {array: self.ds_name})
