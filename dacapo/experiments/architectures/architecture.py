from abc import ABC, abstractmethod

from funlib.geometry import Coordinate


class Architecture(ABC):
    @property
    @abstractmethod
    def input_shape(self):
        """The spatial input shape (i.e., not accounting for channels and batch
        dimensions) of this architecture."""
        pass

    @property
    def eval_shape_increase(self):
        """
        How much to increase the input shape during prediction.
        """
        return Coordinate((0,) * self.input_shape.dims)

    @property
    @abstractmethod
    def num_in_channels(self):
        """Return the number of input channels this architecture expects."""
        pass

    @property
    @abstractmethod
    def num_out_channels(self):
        """Return the number of output channels of this architecture."""
        pass

    @abstractmethod
    def forward(self, x):
        """Process an input tensor."""
        pass

    def scale(self, input_voxel_size: Coordinate) -> Coordinate:
        return input_voxel_size
