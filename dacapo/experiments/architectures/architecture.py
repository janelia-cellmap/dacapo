from abc import ABC, abstractmethod


class Architecture(ABC):

    @property
    @abstractmethod
    def num_in_channels(self):
        """Return the number of input channels this architecture expects."""
        pass
