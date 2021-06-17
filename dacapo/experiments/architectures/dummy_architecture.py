import torch

from .architecture import Architecture


class DummyArchitecture(torch.nn.Module, Architecture):

    def __init__(self, architecture_config):

        super().__init__()

        self.channels_in = architecture_config.num_in_channels
        self.channels_out = architecture_config.num_out_channels

        self.conv = torch.nn.Conv3d(
            self.channels_in,
            self.channels_out,
            kernel_size=3)

    @property
    def num_in_channels(self):
        return self.channels_in

    @property
    def num_out_channels(self):
        return self.channels_out

    def forward(self, x):
        return self.conv(x)
