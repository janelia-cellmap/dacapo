from .arrays import Array

from funlib.geometry import Coordinate

from abc import ABC
from typing import Optional, Any


class Dataset(ABC):
    name: str
    raw: Array
    gt: Optional[Array]
    mask: Optional[Array]
    weight: Optional[int]

    sample_points: Optional[list[Coordinate]]

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, type(self)) and self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self) -> str:
        return f"Dataset({self.name})"

    def __str__(self) -> str:
        return f"Dataset({self.name})"

    def _neuroglancer_layers(self, prefix="", exclude_layers=None):
        layers = {}
        exclude_layers = exclude_layers if exclude_layers is not None else set()
        if (
            self.raw._can_neuroglance()
            and self.raw._source_name() not in exclude_layers
        ):
            layers[self.raw._source_name()] = self.raw._neuroglancer_layer()
        if (
            self.gt is not None
            and self.gt._can_neuroglance()
            and self.gt._source_name() not in exclude_layers
        ):
            layers[self.gt._source_name()] = self.gt._neuroglancer_layer()
        if (
            self.mask is not None
            and self.mask._can_neuroglance()
            and self.mask._source_name() not in exclude_layers
        ):
            layers[self.mask._source_name()] = self.mask._neuroglancer_layer()
        return layers
