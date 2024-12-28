import attr

import torch
from .architecture import ArchitectureConfig

from funlib.geometry import Coordinate


@attr.s
class DummyArchitectureConfig(ArchitectureConfig):
    """
    A dummy architecture configuration class used for testing purposes.
    """

    num_channels_in: int = attr.ib(metadata={"help_text": "Dummy attribute."})

    num_channels_out: int = attr.ib(metadata={"help_text": "Dummy attribute."})

    def module(self) -> torch.nn.Module:
        return torch.nn.Conv3d(
            self.num_in_channels, self.num_out_channels, kernel_size=3
        )

    @property
    def input_shape(self):
        return Coordinate(40, 20, 20)

    @property
    def num_in_channels(self):
        return self.num_channels_in

    @property
    def num_out_channels(self):
        return self.num_channels_out
