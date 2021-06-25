from abc import ABC, abstractmethod


class Architecture(ABC):

    @property
    @abstractmethod
    def input_shape(self):
        """The spatial input shape (i.e., not accounting for channels and batch
        dimensions) of this architecture."""
        pass

    @property
    @abstractmethod
    def num_in_channels(self):
        """Return the number of input channels this architecture expects."""
        pass

    @property
    @abstractmethod
    def num_out_channels(self):
        """Return the number of output channels of this architecture."""
        pass

    @abstractmethod
    def forward(self, x):
        """Process an input tensor."""
        pass
