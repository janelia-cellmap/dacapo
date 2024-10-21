from funlib.geometry import Coordinate
from funlib.persistence import Array
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
    Methods:
        __eq__(other):
            Overloaded equality operator for dataset objects.
        __hash__():
            Calculates a hash for the dataset.
        __repr__():
            Returns the official string representation of the dataset object.
        __str__():
            Returns the string representation of the dataset object.
        _neuroglancer_layers(prefix="", exclude_layers=None):
            Generates neuroglancer layers for raw, gt and mask if they can be viewed by neuroglance, excluding those in
            the exclude_layers.
    Notes:
        This class is a base class and should not be instantiated.
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
        Raises:
            NotImplementedError: If the method is not implemented in the derived class.
        Examples:
            >>> dataset1 = Dataset("dataset1")
            >>> dataset2 = Dataset("dataset2")
            >>> dataset1 == dataset2
            False
        Notes:
            This method is used to compare two dataset objects.
        """
        return isinstance(other, type(self)) and self.name == other.name

    def __hash__(self) -> int:
        """
        Calculates a hash for the dataset.

        Returns:
            int: The hash of the dataset name.
        Raises:
            NotImplementedError: If the method is not implemented in the derived class.
        Examples:
            >>> dataset = Dataset("dataset")
            >>> hash(dataset)
            123456
        Notes:
            This method is used to calculate a hash for the dataset.
        """
        return hash(self.name)

    def __repr__(self) -> str:
        """
        Returns the official string representation of the dataset object.

        Returns:
            str: String representation of the dataset.
        Raises:
            NotImplementedError: If the method is not implemented in the derived class.
        Examples:
            >>> dataset = Dataset("dataset")
            >>> dataset
            Dataset(dataset)
        Notes:
            This method is used to return the official string representation of the dataset object.
        """
        return f"ds_{self.name.replace('/', '_')}"

    def __str__(self) -> str:
        """
        Returns the string representation of the dataset object.

        Args:
            self (Dataset): The dataset object.
        Returns:
            str: String representation of the dataset.
        Raises:
            NotImplementedError: If the method is not implemented in the derived class.
        Examples:
            >>> dataset = Dataset("dataset")
            >>> print(dataset)
            Dataset(dataset)
        Notes:
            This method is used to return the string representation of the dataset object.
        """
        return f"ds_{self.name.replace('/', '_')}"

    def _neuroglancer_layers(self, prefix="", exclude_layers=None):
        """
        Generates neuroglancer layers for raw, gt and mask if they can be viewed by neuroglance, excluding those in
        the exclude_layers.

        Args:
            prefix (str, optional): A prefix to be added to the layer names.
            exclude_layers (set, optional): A set of layer names to exclude.
        Returns:
            dict: A dictionary containing layer names as keys and corresponding neuroglancer layer as values.
        Raises:
            NotImplementedError: If the method is not implemented in the derived class.
        Examples:
            >>> dataset = Dataset("dataset")
            >>> dataset._neuroglancer_layers()
            {"raw": neuroglancer_layer}
        Notes:
            This method is used to generate neuroglancer layers for raw, gt and mask if they can be viewed by neuroglance.
        """
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
