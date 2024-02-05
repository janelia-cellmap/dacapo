# TODO
from .architecture import Architecture

from funlib.geometry import Coordinate

import torch


class DummyArchitecture(Architecture):
    def __init__(self, architecture_config):
        super().__init__()

        self.channels_in = architecture_config.num_in_channels
        self.channels_out = architecture_config.num_out_channels

        self.conv = torch.nn.Conv3d(self.channels_in, self.channels_out, kernel_size=3)

    @property
    def input_shape(self):
        return Coordinate(40, 20, 20)

    @property
    def num_in_channels(self):
        return self.channels_in

    @property
    def num_out_channels(self):
        return self.channels_out

    def forward(self, x):
        return self.conv(x)
