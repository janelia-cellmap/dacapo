from .dummy_datasplit import DummyDataSplit
from .datasplit_config import DataSplitConfig
from .datasets import DatasetConfig, DummyDatasetConfig

import attr

from typing import Tuple


@attr.s
class DummyDataSplitConfig(DataSplitConfig):
    

    # Members with default values
    datasplit_type = DummyDataSplit
    train_config: DatasetConfig = attr.ib(DummyDatasetConfig(name="dummy_dataset"))

    def verify(self) -> Tuple[bool, str]:
        
        return False, "This is a DummyDataSplit and is never valid"
