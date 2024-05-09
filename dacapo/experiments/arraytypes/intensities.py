from .arraytype import ArrayType


import attr

from typing import Dict


@attr.s
class IntensitiesArray(ArrayType):
    """
    An IntensitiesArray is an Array of measured intensities. Each voxel has a value in the range [min, max].

    Attributes:
        channels (Dict[int, str]): A mapping from channel to a name describing that channel.
        min (float): The minimum possible value of your intensities.
        max (float): The maximum possible value of your intensities.
    Methods:
        __attrs_post_init__(self): This method is called after the instance has been initialized by the constructor.
        interpolatable(self) -> bool: It is a method that returns True.
    Note:
        This class is used to create an IntensitiesArray object which is used to represent an array of measured intensities.
    """

    channels: Dict[int, str] = attr.ib(
        metadata={
            "help_text": "A mapping from channel to a name describing that channel."
        }
    )
    min: float = attr.ib(
        metadata={"help_text": "The minimum possible value of your intensities."}
    )
    max: float = attr.ib(
        metadata={"help_text": "The maximum possible value of your intensities."}
    )

    @property
    def interpolatable(self) -> bool:
        """
        Method to return True.

        Returns:
            bool
                Returns a boolean value of True representing that the values are interpolatable.
        Raises:
            NotImplementedError
                This method is not implemented in this class.
        Examples:
            >>> intensities_array = IntensitiesArray(channels={1: "channel1"}, min=0, max=1)
            >>> intensities_array.interpolatable
            True
        Note:
            This method is used to check if the array is interpolatable.
        """
        return True
