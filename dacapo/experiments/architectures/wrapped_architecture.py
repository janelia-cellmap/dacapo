import attr

import torch
from .architecture import ArchitectureConfig

from funlib.geometry import Coordinate


@attr.s
class WrappedArchitectureConfig(ArchitectureConfig):
    """
    A thin wrapper allowing users to pass in any architecture they want
    """

    module_: torch.nn.Module = attr.ib(
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

    input_shape_: Coordinate = attr.ib(
        metadata={
            "help_text": "The input shape spatial dimensions (t,z,y,x). "
            "Does not include batch or channel dimension shapes"
        }
    )

    scale_: Coordinate | None = attr.ib(
        default=None,
        metadata={
            "help_text": "The amount by which to scale each dimension in case of up or down scaling networks"
        },
    )

    def module(self) -> torch.nn.Module:
        return self.module_

    @property
    def input_shape(self):
        return self.input_shape_

    @property
    def num_in_channels(self):
        return self.fmaps_in

    @property
    def num_out_channels(self):
        return self.fmaps_out

    def scale(self, input_voxel_size: Coordinate) -> Coordinate:
        if self.scale_ is not None:
            return input_voxel_size // self.scale_
        else:
            return input_voxel_size
