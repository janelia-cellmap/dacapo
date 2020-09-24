from abc import ABC, abstractmethod


class ArrayDataset(ABC):
    """
    A class representing a dataset of arrays used for training, validation, or testing.
    i.e. Raw or Labels or Mask etc.

    Must have some attributes such as voxel_size, spatial_dims, etc.
    Must have a get_source function that takes as input a gunpowder Array or Graph
    key, and then provides a gunpowder source node from which that array
    or graph can be requested.
    """

    @property
    @abstractmethod
    def voxel_size(self):
        pass

    @property
    @abstractmethod
    def spatial_dims(self):
        pass

    @property
    @abstractmethod
    def offset(self):
        pass

    @property
    @abstractmethod
    def shape(self):
        pass

    @property
    @abstractmethod
    def spatial_shape(self):
        pass

    @property
    @abstractmethod
    def roi(self):
        pass

    @property
    @abstractmethod
    def axes(self):
        pass

    @property
    @abstractmethod
    def num_channels(self):
        pass

    @property
    @abstractmethod
    def num_samples(self):
        pass

    @property
    @abstractmethod
    def num_classes(self):
        pass

    @property
    @abstractmethod
    def background_label(self):
        pass

    @abstractmethod
    def get_source(self):
        pass


class GraphDataset(ABC):
    """
    A class representing a graph dataset used for training, validation, or testing.

    Must have some attributes such as voxel_size, spatial_dims, etc.
    Must have a get_source function that takes as input a gunpowder Array or Graph
    key, and then provides a gunpowder source node from which that array
    or graph can be requested.
    """

    @property
    @abstractmethod
    def spatial_dims(self):
        pass

    @property
    @abstractmethod
    def roi(self):
        pass

    @property
    @abstractmethod
    def axes(self):
        pass

    @abstractmethod
    def get_source(self):
        pass
