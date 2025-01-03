import attr

import torch
from .architecture import ArchitectureConfig

from funlib.geometry import Coordinate


@attr.s
class WrappedArchitectureConfig(ArchitectureConfig):
    """
    A thin wrapper allowing users to pass in any architecture they want
    """

    _module: torch.nn.Module = attr.ib(
        metadata={"help_text": "The `torch.nn.Module` you would like to use"}
    )

    fmaps_in: int = attr.ib(
        metadata={"help_text": "The number of channels that your model takes as input"}
    )

    fmaps_out: int = attr.ib(
        metadata={
            "help_text": "The number of channels that your model generates as output"
        }
    )

    _input_shape: Coordinate = attr.ib(
        metadata={
            "help_text": "The input shape spatial dimensions (t,z,y,x). "
            "Does not include batch or channel dimension shapes"
        }
    )

    _scale: Coordinate | None = attr.ib(
        default=None,
        metadata={
            "help_text": "The amount by which to scale each dimension in case of up or down scaling networks"
        },
    )

    def module(self) -> torch.nn.Module:
        return self._module

    @property
    def input_shape(self):
        return Coordinate(self._input_shape)

    @property
    def num_in_channels(self):
        return self.fmaps_in

    @property
    def num_out_channels(self):
        return self.fmaps_out

    def scale(self, input_voxel_size: Coordinate) -> Coordinate:
        if self._scale is not None:
            return input_voxel_size // self._scale
        else:
            return input_voxel_size
