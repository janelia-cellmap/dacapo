from .dummy_datasplit import DummyDataSplit
from .datasplit_config import DataSplitConfig
from .datasets import DatasetConfig, DummyDatasetConfig

import attr

from typing import List, Tuple


@attr.s
class DummyDataSplitConfig(DataSplitConfig):
    """This is just a dummy DataSplit config used for testing. None of the
    attributes have any particular meaning."""

    datasplit_type = DummyDataSplit

    train_config: DatasetConfig = attr.ib(DummyDatasetConfig(name="dummy_dataset"))

    def verify(self) -> Tuple[bool, str]:
        return False, "This is a DummyDataSplit and is never valid"
