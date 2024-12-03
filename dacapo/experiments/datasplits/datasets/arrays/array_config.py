import attr

from typing import Tuple
from abc import ABC, abstractmethod
from funlib.persistence import Array


@attr.s
class ArrayConfig(ABC):
    """
    Base class for array configurations. Each subclass of an
    `Array` should have a corresponding config class derived from
    `ArrayConfig`. This class should be used to store the configuration
    of the array.

    Attributes:
        name (str): A unique name for this array. This will be saved so you
            and others can find and reuse this array. Keep it short
            and avoid special characters.
    Methods:
        verify(self) -> Tuple[bool, str]: This method is used to check whether this is a valid Array.
    Note:
        This class is used to create a base class for array configurations. Each subclass of an
        `Array` should have a corresponding config class derived from `ArrayConfig`.
        This class should be used to store the configuration of the array.
    """

    name: str = attr.ib(
        metadata={
            "help_text": "A unique name for this array. This will be saved so you "
            "and others can find and reuse this array. Keep it short "
            "and avoid special characters."
        }
    )

    @abstractmethod
    def array(self, mode: str = "r") -> Array:
        pass

    def verify(self) -> Tuple[bool, str]:
        """
        Check whether this is a valid Array

        Returns:
            Tuple[bool, str]: A tuple with the first element being a boolean
            indicating whether the array is valid and the second element being
            a string with a message explaining why the array is invalid
        Raises:
            NotImplementedError: This method is not implemented in this class
        Examples:
            >>> array_config = ArrayConfig(name="array_config")
            >>> array_config.verify()
            (True, "No validation for this Array")
        Note:
            This method is used to check whether this is a valid Array.
        """
        return True, "No validation for this Array"
