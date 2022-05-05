import attr

from .dummy_architecture import DummyArchitecture
from .architecture_config import ArchitectureConfig

from typing import Tuple

@attr.s
class DummyArchitectureConfig(ArchitectureConfig):
    """This is just a dummy architecture config used for testing. None of the
    attributes have any particular meaning."""

    architecture_type = DummyArchitecture

    num_in_channels: int = attr.ib(metadata={"help_text": "Dummy attribute."})

    num_out_channels: int = attr.ib(metadata={"help_text": "Dummy attribute."})

    def verify(self) -> Tuple[bool, str]:
        return False, "This is a DummyArchitectureConfig and is never valid"
