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

    def _neuroglancer_layers(self, prefix=""):
        layers = {}
        if self.raw._can_neuroglance():
            layers[self.raw._source_name()] = self.raw._neuroglancer_layer()
        if self.gt is not None and self.gt._can_neuroglance():
            layers[self.gt._source_name()] = self.gt._neuroglancer_layer()
        if self.mask is not None and self.mask._can_neuroglance():
            layers[self.mask._source_name()] = self.mask._neuroglancer_layer()
        return layers
