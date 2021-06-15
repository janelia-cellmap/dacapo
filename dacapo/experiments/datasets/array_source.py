from abc import ABC, abstractmethod


class ArraySource(ABC):

    @property
    @abstractmethod
    def axes(self):
        """Returns the axes of this dataset as a string of charactes, as they
        are indexed. Permitted characters are:

            * ``zyx`` for spatial dimensions
            * ``c`` for channels
            * ``s`` for samples
        """
        pass

    @property
    @abstractmethod
    def dims(self):
        """Returns the number of spatial dimensions."""
        pass
