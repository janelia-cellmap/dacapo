from .arrays import Array

from abc import ABC, abstractmethod


class Dataset(ABC):

    @property
    @abstractmethod
    def raw(self) -> Array:
        """The Dataset to train on."""
        pass
