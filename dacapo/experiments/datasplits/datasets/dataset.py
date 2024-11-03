from .arrays import Array
from funlib.geometry import Coordinate
from abc import ABC
from typing import Optional, Any, List


class Dataset(ABC):
    

    name: str
    raw: Array
    gt: Optional[Array]
    mask: Optional[Array]
    weight: Optional[int]
    sample_points: Optional[List[Coordinate]]

    def __eq__(self, other: Any) -> bool:
        
        return isinstance(other, type(self)) and self.name == other.name

    def __hash__(self) -> int:
        
        return hash(self.name)

    def __repr__(self) -> str:
        
        return f"ds_{self.name.replace('/', '_')}"

    def __str__(self) -> str:
        
        return f"ds_{self.name.replace('/', '_')}"

    def _neuroglancer_layers(self, prefix="", exclude_layers=None):
        
        layers = {}
        exclude_layers = exclude_layers if exclude_layers is not None else set()
        if (
            self.raw._can_neuroglance()
            and self.raw._source_name() not in exclude_layers
        ):
            layers[self.raw._source_name()] = self.raw._neuroglancer_layer()
        if self.gt is not None and self.gt._can_neuroglance():
            new_layers = self.gt._neuroglancer_layer()
            if isinstance(new_layers, list):
                names = self.gt._source_name()
                for name, layer in zip(names, new_layers):
                    if name not in exclude_layers:
                        layers[name] = layer
            elif self.gt._source_name() not in exclude_layers:
                layers[self.gt._source_name()] = new_layers
        if self.mask is not None and self.mask._can_neuroglance():
            new_layers = self.mask._neuroglancer_layer()
            if isinstance(new_layers, list):
                names = self.mask._source_name()
                for name, layer in zip(names, new_layers):
                    if name not in exclude_layers:
                        layers[name] = layer
            elif self.gt._source_name() not in exclude_layers:
                layers["mask_" + self.mask._source_name()] = new_layers
        return layers
