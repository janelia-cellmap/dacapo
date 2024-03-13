from funlib.geometry import Coordinate

import torch

from abc import ABC, abstractmethod


class Architecture(torch.nn.Module, ABC):
    """
    An abstract base class for defining the architecture of a neural network model.
    It is inherited from PyTorch's Module and built-in class `ABC` (Abstract Base Classes).
    Other classes can inherit this class to define their own specific variations of architecture.
    It requires to implement several property methods, and also includes additional methods related to the architecture design.
    """

    @property
    @abstractmethod
    def input_shape(self) -> Coordinate:
        """
        Abstract method to define the spatial input shape for the neural network architecture.
        The shape should not account for the channels and batch dimensions.

        Returns:
            Coordinate: The spatial input shape.
        """
        pass

    @property
    def eval_shape_increase(self) -> Coordinate:
        """
        Provides information about how much to increase the input shape during prediction.

        Returns:
            Coordinate: An instance representing the amount to increase in each dimension of the input shape.
        """
        return Coordinate((0,) * self.input_shape.dims)

    @property
    @abstractmethod
    def num_in_channels(self) -> int:
        """
        Abstract method to return number of input channels required by the architecture.

        Returns:
            int: Required number of input channels.
        """
        pass

    @property
    @abstractmethod
    def num_out_channels(self) -> int:
        """
        Abstract method to return the number of output channels provided by the architecture.

        Returns:
            int: Number of output channels.
        """
        pass

    @property
    def dims(self) -> int:
        """
        Returns the number of dimensions of the input shape.

        Returns:
            int: The number of dimensions.
        """
        return self.input_shape.dims

    def scale(self, input_voxel_size: Coordinate) -> Coordinate:
        """
        Method to scale the input voxel size as required by the architecture.

        Args:
            input_voxel_size (Coordinate): The original size of the input voxel.

        Returns:
            Coordinate: The scaled voxel size.
        """
        return input_voxel_size
