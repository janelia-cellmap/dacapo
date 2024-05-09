import attr

from .array_config import ArrayConfig
from .dummy_array import DummyArray

from typing import Tuple


@attr.s
class DummyArrayConfig(ArrayConfig):
    """
    This is just a dummy array  config used for testing. None of the
    attributes have any particular meaning. It is used to test the
    ArrayConfig class.

    Methods:
        to_array: Returns the DummyArray object
        verify: Returns whether the DummyArrayConfig is valid
    Notes:
        The source_array_config must be an ArrayConfig object.

    """

    array_type = DummyArray

    def verify(self) -> Tuple[bool, str]:
        """
        Check whether this is a valid Array

        Returns:
            Tuple[bool, str]: Whether the Array is valid and a message
        Raises:
            ValueError: If the source is not a tuple of strings
        Examples:
            >>> dummy_array_config = DummyArrayConfig(...)
            >>> dummy_array_config.verify()
            (False, "This is a DummyArrayConfig and is never valid")
        Notes:
            The source must be a tuple of strings.
        """
        return False, "This is a DummyArrayConfig and is never valid"
