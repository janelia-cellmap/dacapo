import attr

from .array_config import ArrayConfig


from typing import Tuple


@attr.s
class DVIDArrayConfig(ArrayConfig):
    """
    This config class provides the necessary configuration for a DVID array. It takes a source string and returns the DVIDArray object.

    Attributes:
        source (Tuple[str, str, str]): The source strings
    Methods:
        to_array: Returns the DVIDArray object
    Notes:
        The source must be a tuple of strings.

    """

    source: Tuple[str, str, str] = attr.ib(
        metadata={"help_text": "The source strings."}
    )

    def array(self, mode: str = "r"):
        raise NotImplementedError

    def verify(self) -> Tuple[bool, str]:
        """
        Check whether this is a valid Array

        Returns:
            Tuple[bool, str]: Whether the Array is valid and a message
        Raises:
            ValueError: If the source is not a tuple of strings
        Examples:
            >>> dvid_array_config = DVIDArrayConfig(...)
            >>> dvid_array_config.verify()
            (True, "No validation for this Array")
        Notes:
            The source must be a tuple of strings.
        """
        return True, "No validation for this Array"
