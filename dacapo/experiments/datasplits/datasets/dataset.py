from .arraystores import ArrayStore

from abc import ABC, abstractmethod


class Dataset(ABC):

    @property
    @abstractmethod
    def raw(self) -> ArrayStore:
        """The Dataset to train on."""
        pass
