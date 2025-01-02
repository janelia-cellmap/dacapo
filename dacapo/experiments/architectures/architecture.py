import attr

from funlib.geometry import Coordinate
import torch

from abc import ABC, abstractmethod


@attr.s
class ArchitectureConfig(ABC):
    """
    A base class for an configurable architecture that can be used in DaCapo

    Attributes:
        name : str
            a unique name for the architecture.
    Methods:
        verify()
            validates the given architecture.
    Note:
        The class is abstract and requires to implement the abstract methods.
    """

    name: str = attr.ib(
        metadata={
            "help_text": "A unique name for this architecture. This will be saved so "
            "you and others can find and reuse this task. Keep it short "
            "and avoid special characters."
        }
    )

    @abstractmethod
    def module(self) -> torch.nn.Module:
        pass


    @property
    @abstractmethod
    def input_shape(self) -> Coordinate:
        """
        Abstract method to define the spatial input shape for the neural network architecture.
        The shape should not account for the channels and batch dimensions.
        """
        pass

    @property
    def dims(self) -> int:
        return self.input_shape.dims

    @property
    def eval_shape_increase(self) -> Coordinate:
        """
        Provides information about how much to increase the input shape during prediction.
        """
        return Coordinate((0,) * self.dims)

    @property
    @abstractmethod
    def num_in_channels(self) -> int:
        """
        Abstract method to return number of input channels required by the architecture.
        """
        pass

    @property
    @abstractmethod
    def num_out_channels(self) -> int:
        """
        Abstract method to return the number of output channels provided by the architecture.
        """
        pass

    def scale(self, input_voxel_size: Coordinate) -> Coordinate:
        """
        Method to scale the input voxel size as required by the architecture.
        """
        return input_voxel_size