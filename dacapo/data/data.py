from .dataset import Dataset


class RawData:

    def __init__(self, filename, train_ds, validate_ds):

        self.train = Dataset(filename, train_ds)
        self.validate = Dataset(filename, validate_ds)

        assert self.train.num_channels == self.validate.num_channels
        assert self.train.voxel_size == self.validate.voxel_size
        assert self.train.spatial_dims == self.validate.spatial_dims

        self.num_channels = self.train.num_channels
        self.voxel_size = self.train.voxel_size
        self.spatial_dims = self.train.spatial_dims


class GtData:

    def __init__(self, filename, train_ds, validate_ds):

        self.train = Dataset(filename, train_ds)
        self.validate = Dataset(filename, validate_ds)

        assert self.train.num_channels == self.validate.num_channels
        assert self.train.voxel_size == self.validate.voxel_size
        assert self.train.num_classes == self.validate.num_classes
        assert self.train.background_label == self.validate.background_label
        assert self.train.spatial_dims == self.validate.spatial_dims

        self.num_channels = self.train.num_channels
        self.voxel_size = self.train.voxel_size
        self.num_classes = self.train.num_classes
        self.background_label = self.train.background_label
        self.spatial_dims = self.train.spatial_dims


class Data:

    def __init__(self, data_config):

        self.filename = str(data_config.filename)
        self.raw = RawData(
            self.filename,
            data_config.train_raw,
            data_config.validate_raw)
        self.gt = GtData(
            self.filename,
            data_config.train_gt,
            data_config.validate_gt)

        assert self.raw.spatial_dims == self.gt.spatial_dims
