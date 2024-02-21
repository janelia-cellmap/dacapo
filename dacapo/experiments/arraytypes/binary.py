from .arraytype import ArrayType

import attr

from typing import Dict


@attr.s
class BinaryArray(ArrayType):
    """
    A subclass of ArrayType representing BinaryArray. The BinaryArray object is created with two attributes; channels.
    Each voxel in this array is either 1 or 0.

    Attributes:
        channels (Dict[int, str]): A dictionary attribute representing channel mapping with its binary classification.

    Args:
        channels (Dict[int, str]): A dictionary input where keys are channel numbers and values are their corresponding class for binary classification.

    Methods:
        interpolatable: Returns False as binary array type is not interpolatable.
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
        """
        return False