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
        num_in_channels(): Returns the number of input channels for this architecture.
        num_out_channels(): Returns the number of output channels for this architecture.
    Note:
        This class is used to represent a dummy architecture layer for a 3D CNN.
    """

    def __init__(self, architecture_config):
        """
        Constructor for the DummyArchitecture class. Initializes the 3D convolution object.

        Args:
            architecture_config: An architecture configuration object.
        Raises:
            NotImplementedError: This method is not implemented in this class.
        Examples:
            >>> architecture_config = ArchitectureConfig(num_in_channels=1, num_out_channels=1)
            >>> dummy_architecture = DummyArchitecture(architecture_config)
        Note:
            This method is used to initialize the DummyArchitecture class.
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
        Raises:
            NotImplementedError: This method is not implemented in this class.
        Examples:
            >>> dummy_architecture.input_shape
            Coordinate(x=40, y=20, z=20)
        Note:
            This method is used to return the input shape for this architecture.
        """
        return Coordinate(40, 20, 20)

    @property
    def num_in_channels(self):
        """
        Returns the number of input channels for this architecture.

        Returns:
            int: Number of input channels.
        Raises:
            NotImplementedError: This method is not implemented in this class.
        Examples:
            >>> dummy_architecture.num_in_channels
            1
        Note:
            This method is used to return the number of input channels for this architecture.
        """
        return self.channels_in

    @property
    def num_out_channels(self):
        """
        Returns the number of output channels for this architecture.

        Returns:
            int: Number of output channels.
        Raises:
            NotImplementedError: This method is not implemented in this class.
        Examples:
            >>> dummy_architecture.num_out_channels
            1
        Note:
            This method is used to return the number of output channels for this architecture.
        """
        return self.channels_out

    def forward(self, x):
        """
        Perform the forward pass of the network.

        Args:
            x: Input tensor.
        Returns:
            Tensor: Output tensor after the forward pass.
        Raises:
            NotImplementedError: This method is not implemented in this class.
        Examples:
            >>> dummy_architecture = DummyArchitecture(architecture_config)
            >>> x = torch.randn(1, 1, 40, 20, 20)
            >>> dummy_architecture.forward(x)
        Note:
            This method is used to perform the forward pass of the network.
        """
        return self.conv(x)
