from .dummy_dataset import DummyDataset
from .dataset_config import DatasetConfig
from .arrays import ArrayConfig, DummyArrayConfig

import attr


@attr.s
class DummyDatasetConfig(DatasetConfig):
    """This is just a dummy DataSplit config used for testing. None of the
    attributes have any particular meaning."""

    datasplit_type = DummyDataset

    raw_config: ArrayConfig = attr.ib(DummyArrayConfig(name="dummy_array"))
    gt_config: ArrayConfig = attr.ib(DummyArrayConfig(name="dummy_gt_array"))
