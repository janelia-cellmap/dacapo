from .arraytype import ArrayType

import attr

from typing import Dict


@attr.s
class BinaryArray(ArrayType):
    """
    A subclass of ArrayType representing BinaryArray. The BinaryArray object is created with two attributes; channels.
    Each voxel in this array is either 1 or 0. The class of each voxel can be determined by simply taking the argmax.

    Attributes:
        channels (Dict[int, str]): A dictionary attribute representing channel mapping with its binary classification.
    Methods:
        interpolatable: Returns False as binary array type is not interpolatable.
    Note:
        This class is used to represent a BinaryArray object in the system.
    """

    channels: Dict[int, str] = attr.ib(
        metadata={
            "help_text": "A mapping from channel to class for the binary classification."
        }
    )

    @property
    def interpolatable(self) -> bool:
        """
        This function returns the interpolatable property value of the binary array.

        Returns:
            bool: Always returns False because interpolation is not possible.
        Raises:
            NotImplementedError: This method is not implemented in this class.
        Examples:
            >>> binary_array = BinaryArray(channels={1: "class1"})
            >>> binary_array.interpolatable
            False
        Note:
            This method is used to check if the array is interpolatable.
        """
        return False
