import attr

from .array_source_config import ArraySourceConfig
from .dummy_array_source import DummyArraySource


@attr.s
class DummyArraySourceConfig(ArraySourceConfig):
    """This is just a dummy array source config used for testing. None of the
    attributes have any particular meaning."""

    source_type = DummyArraySource

    filename: str = attr.ib(
        metadata={
            "help_text":
                "Dummy attribute."
        }
    )
