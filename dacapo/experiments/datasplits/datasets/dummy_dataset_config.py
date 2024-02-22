from .dummy_dataset import DummyDataset
from .dataset_config import DatasetConfig
from .arrays import ArrayConfig, DummyArrayConfig

import attr

from typing import Tuple


@attr.s
class DummyDatasetConfig(DatasetConfig):
    """This is just a dummy DataSplit config used for testing. None of the
    attributes have any particular meaning."""

    dataset_type = DummyDataset

    raw_config: ArrayConfig = attr.ib(DummyArrayConfig(name="dummy_array"))

    def verify(self) -> Tuple[bool, str]:
        return False, "This is a DummyDatasetConfig and is never valid"
