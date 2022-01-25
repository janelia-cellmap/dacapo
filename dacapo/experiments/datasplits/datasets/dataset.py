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

    def _neuroglancer_layers(self, prefix="", exclude_layers=None):
        layers = {}
        exclude_layers = exclude_layers if exclude_layers is not None else set()
        if (
            self.raw._can_neuroglance()
            and not self.raw._source_name() in exclude_layers
        ):
            layers[self.raw._source_name()] = self.raw._neuroglancer_layer()
        if (
            self.gt is not None
            and self.gt._can_neuroglance()
            and not self.gt._source_name() in exclude_layers
        ):
            layers[self.gt._source_name()] = self.gt._neuroglancer_layer()
        if (
            self.mask is not None
            and self.mask._can_neuroglance()
            and not self.mask._source_name() in exclude_layers
        ):
            layers[self.mask._source_name()] = self.mask._neuroglancer_layer()
        return layers
