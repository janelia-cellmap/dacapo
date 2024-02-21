"""
This module implements dummy architecture layer for a 3D convolutional neural network.

Classes:
    DummyArchitecture(Architecture)
"""

from .architecture import Architecture
from funlib.geometry import Coordinate
import torch


class DummyArchitecture(Architecture):
    """
    A class used to represent a dummy architecture layer for a 3D CNN.

    Attributes:
        channels_in: An integer representing the number of input channels.
        channels_out: An integer representing the number of output channels.
        conv: A 3D convolution object.
        input_shape: A coordinate object representing the shape of the input.

    Methods:
        forward(x): Performs the forward pass of the network.
    """

    def __init__(self, architecture_config):
        """
        Args:
            architecture_config: An object containing the configuration settings for the architecture.
        """
        super().__init__()

        self.channels_in = architecture_config.num_in_channels
        self.channels_out = architecture_config.num_out_channels

        self.conv = torch.nn.Conv3d(self.channels_in, self.channels_out, kernel_size=3)

    @property
    def input_shape(self):
        """
        Returns the input shape for this architecture.

        Returns:
            Coordinate: Input shape of the architecture.
        """
        return Coordinate(40, 20, 20)

    @property
    def num_in_channels(self):
        """
        Returns the number of input channels for this architecture.

        Returns:
            int: Number of input channels.
        """
        return self.channels_in

    @property
    def num_out_channels(self):
        """
        Returns the number of output channels for this architecture.

        Returns:
            int: Number of output channels.
        """
        return self.channels_out

    def forward(self, x):
        """
        Perform the forward pass of the network.

        Args:
            x: Input tensor.

        Returns:
            Tensor: Output tensor after the forward pass.
        """
        return self.conv(x)
