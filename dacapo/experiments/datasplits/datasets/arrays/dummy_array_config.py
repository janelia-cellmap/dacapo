import attr

from .array_config import ArrayConfig
from .dummy_array import DummyArray

from typing import Tuple


@attr.s
class DummyArrayConfig(ArrayConfig):
    """This is just a dummy array  config used for testing. None of the
    attributes have any particular meaning."""

    array_type = DummyArray

    def verify(self) -> Tuple[bool, str]:
        return False, "This is a DummyArrayConfig and is never valid"
