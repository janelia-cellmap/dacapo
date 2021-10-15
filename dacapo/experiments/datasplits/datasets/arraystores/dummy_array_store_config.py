import attr

from .array_store_config import ArrayStoreConfig
from .dummy_array_store import DummyArrayStore


@attr.s
class DummyArrayStoreConfig(ArrayStoreConfig):
    """This is just a dummy array store config used for testing. None of the
    attributes have any particular meaning."""

    array_store_type = DummyArrayStore
