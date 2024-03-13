from .arrays import Array
from funlib.geometry import Coordinate
from abc import ABC
from typing import Optional, Any, List


class Dataset(ABC):
    """
    A class to represent a dataset.

    Attributes:
        name (str): The name of the dataset.
        raw (Array): The raw dataset.
        gt (Array, optional): The ground truth data.
        mask (Array, optional): The mask for the data.
        weight (int, optional): The weight of the dataset.
        sample_points (list[Coordinate], optional): The list of sample points in the dataset.
    """

    name: str
    raw: Array
    gt: Optional[Array]
    mask: Optional[Array]
    weight: Optional[int]
    sample_points: Optional[List[Coordinate]]

    def __eq__(self, other: Any) -> bool:
        """
        Overloaded equality operator for dataset objects.

        Args:
            other (Any): The object to compare with the dataset.

        Returns:
            bool: True if the object is also a dataset and they have the same name, False otherwise.
        """
        return isinstance(other, type(self)) and self.name == other.name

    def __hash__(self) -> int:
        """
        Calculates a hash for the dataset.

        Returns:
            int: The hash of the dataset name.
        """
        return hash(self.name)

    def __repr__(self) -> str:
        """
        Returns the official string representation of the dataset object.

        Returns:
            str: String representation of the dataset.
        """
        return f"Dataset({self.name})"

    def __str__(self) -> str:
        """
        Returns the string representation of the dataset object.

        Returns:
            str: String representation of the dataset.
        """
        return f"Dataset({self.name})"

    def _neuroglancer_layers(self, prefix="", exclude_layers=None):
        """
        Generates neuroglancer layers for raw, gt and mask if they can be viewed by neuroglance, excluding those in
        the exclude_layers.

        Args:
            prefix (str, optional): A prefix to be added to the layer names.
            exclude_layers (set, optional): A set of layer names to exclude.

        Returns:
            dict: A dictionary containing layer names as keys and corresponding neuroglancer layer as values.
        """
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
