from .arrays import Array

from abc import ABC, abstractmethod
from typing import Optional


class Dataset(ABC):

    @property
    @abstractmethod
    def raw(self) -> Array:
        """The Input data to your model."""
        pass

    @property
    def gt(self) -> Optional[Array]:
        """The GT Array for `raw`"""
        return None

    @property
    def mask(self) -> Optional[Array]:
        """The Mask Array for `raw`"""
        return None