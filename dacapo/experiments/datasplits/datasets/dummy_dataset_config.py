from .dummy_dataset import DummyDataset
from .dataset_config import DatasetConfig
from .arraystores import ArrayStoreConfig, DummyArrayStoreConfig

import attr


@attr.s
class DummyDatasetConfig(DatasetConfig):
    """This is just a dummy DataSplit config used for testing. None of the
    attributes have any particular meaning."""

    datasplit_type = DummyDataset

    raw_config: ArrayStoreConfig = attr.ib(DummyArrayStoreConfig(name="dummy_array"))
