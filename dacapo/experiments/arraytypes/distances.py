from .arraytype import ArrayType

import attr

from typing import Dict


@attr.s
class DistanceArray(ArrayType):
    """
    An array containing signed distances to the nearest boundary voxel for a particular label class.
    Distances should be positive outside an object and negative inside an object. The distance should be 0 on the boundary.
    The class of each voxel can be determined by simply taking the argmin. The distance should be in the range [-max, max].

    Attributes:
        classes (Dict[int, str]): A mapping from channel to class on which distances were calculated.
        max (float): The maximum possible distance value of your distances.
    Methods:
        interpolatable(self) -> bool: It is a method that returns True.
    Note:
        This class is used to create a DistanceArray object which is used to represent an array containing signed distances to the nearest boundary voxel for a particular label class.
        The class of each voxel can be determined by simply taking the argmin.
    """

    classes: Dict[int, str] = attr.ib(
        metadata={
            "help_text": "A mapping from channel to class on which distances were calculated"
        }
    )

    @property
    def interpolatable(self) -> bool:
        """
        Checks if the array is interpolatable. Returns True for this class.

        Returns:
            bool: True indicating that the data can be interpolated.
        Raises:
            NotImplementedError: This method is not implemented in this class
        Examples:
            >>> distance_array = DistanceArray(classes={1: "class1"})
            >>> distance_array.interpolatable
            True
        Note:
            This method is used to check if the array is interpolatable.
        """
        return True
