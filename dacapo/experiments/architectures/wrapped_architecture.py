import attr
import re

import torch
from .architecture import ArchitectureConfig

from funlib.geometry import Coordinate

from pathlib import Path


@attr.s
class WrappedArchitectureConfig(ArchitectureConfig):
    """
    A thin wrapper around user provided `torch.nn.Module` instances. It can be provided
    via a python object, or a file path that can be loaded with `torch.load`. This includes
    pickled modules or jit-compiled modules.
    """

    _module: torch.nn.Module | Path = attr.ib(
        metadata={
            "help_text": "The `torch.nn.Module` you would like to use, or a path to a "
            "pickled or jit-compiled module"
        }
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

    trainable_parameters: str | None = attr.ib(default=None)

    def module(self) -> torch.nn.Module:
        if isinstance(self._module, torch.nn.Module):
            module = self._module
        else:
            module = torch.load(self._module)

        for name, param in module.named_parameters():
            if self.trainable_parameters is not None and re.match(
                self.trainable_parameters, name
            ):
                param.requires_grad = True
            else:
                param.requires_grad = False
        return module

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
