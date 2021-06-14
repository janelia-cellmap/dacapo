from gunpowder import ArrayKey, GraphKey

from abc import ABC, abstractmethod
from typing import Union


class ArrayDatasetABC(ABC):
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
    def axes(self):
        """
        Every array in dacapo is expected to have labelled axes.

        Reserved labels:
            "c": Channel dimension
        """
        pass

    @abstractmethod
    def get_source(self, *output_keys: ArrayKey):
        """
        A dacapo dataset is expected to provide a gunpowder tree that
        provides the given output keys.
        """
        pass


class GraphDatasetABC(ABC):
    """
    A class representing a graph dataset used for training, validation, or testing.

    Must have some attributes such as voxel_size, spatial_dims, etc.
    Must have a get_source function that takes as input a gunpowder Array or Graph
    key, and then provides a gunpowder source node from which that array
    or graph can be requested.
    """

    @abstractmethod
    def get_source(self, *output_keys: GraphKey):
        pass
