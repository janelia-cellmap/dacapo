from funlib.geometry import Coordinate

import torch

from abc import ABC, abstractmethod


class Architecture(torch.nn.Module, ABC):
    """
    An abstract base class for defining the architecture of a neural network model.
    It is inherited from PyTorch's Module and built-in class `ABC` (Abstract Base Classes).
    Other classes can inherit this class to define their own specific variations of architecture.
    It requires to implement several property methods, and also includes additional methods related to the architecture design.

    Attributes:
        input_shape (Coordinate): The spatial input shape for the neural network architecture.
        eval_shape_increase (Coordinate): The amount to increase the input shape during prediction.
        num_in_channels (int): The number of input channels required by the architecture.
        num_out_channels (int): The number of output channels provided by the architecture.
    Methods:
        dims: Returns the number of dimensions of the input shape.
        scale: Scales the input voxel size as required by the architecture.
    Note:
        The class is abstract and requires to implement the abstract methods.
    """

    @property
    @abstractmethod
    def input_shape(self) -> Coordinate:
        """
        Abstract method to define the spatial input shape for the neural network architecture.
        The shape should not account for the channels and batch dimensions.

        Returns:
            Coordinate: The spatial input shape.
        Raises:
            NotImplementedError: If the method is not implemented in the derived class.
        Examples:
            >>> input_shape = Coordinate((128, 128, 128))
            >>> model = MyModel(input_shape)
        Note:
            The method should be implemented in the derived class.

        """
        pass

    @property
    def eval_shape_increase(self) -> Coordinate:
        """
        Provides information about how much to increase the input shape during prediction.

        Returns:
            Coordinate: An instance representing the amount to increase in each dimension of the input shape.
        Raises:
            NotImplementedError: If the method is not implemented in the derived class.
        Examples:
            >>> eval_shape_increase = Coordinate((0, 0, 0))
            >>> model = MyModel(input_shape, eval_shape_increase)
        Note:
            The method is optional and can be overridden in the derived class.
        """
        return Coordinate((0,) * self.input_shape.dims)

    @property
    @abstractmethod
    def num_in_channels(self) -> int:
        """
        Abstract method to return number of input channels required by the architecture.

        Returns:
            int: Required number of input channels.
        Raises:
            NotImplementedError: If the method is not implemented in the derived class.
        Examples:
            >>> num_in_channels = 1
            >>> model = MyModel(input_shape, num_in_channels)
        Note:
            The method should be implemented in the derived class.
        """
        pass

    @property
    @abstractmethod
    def num_out_channels(self) -> int:
        """
        Abstract method to return the number of output channels provided by the architecture.

        Returns:
            int: Number of output channels.
        Raises:
            NotImplementedError: If the method is not implemented in the derived class.
        Examples:
            >>> num_out_channels = 1
            >>> model = MyModel(input_shape, num_out_channels)
        Note:
            The method should be implemented in the derived class.

        """
        pass

    @property
    def dims(self) -> int:
        """
        Returns the number of dimensions of the input shape.

        Returns:
            int: The number of dimensions.
        Raises:
            NotImplementedError: If the method is not implemented in the derived class.
        Examples:
            >>> input_shape = Coordinate((128, 128, 128))
            >>> model = MyModel(input_shape)
            >>> model.dims
            3
        Note:
            The method is optional and can be overridden in the derived class.
        """
        return self.input_shape.dims

    def scale(self, input_voxel_size: Coordinate) -> Coordinate:
        """
        Method to scale the input voxel size as required by the architecture.

        Args:
            input_voxel_size (Coordinate): The original size of the input voxel.
        Returns:
            Coordinate: The scaled voxel size.
        Raises:
            NotImplementedError: If the method is not implemented in the derived class.
        Examples:
            >>> input_voxel_size = Coordinate((1, 1, 1))
            >>> model = MyModel(input_shape)
            >>> model.scale(input_voxel_size)
            Coordinate((1, 1, 1))
        Note:
            The method is optional and can be overridden in the derived class.
        """
        return input_voxel_size
