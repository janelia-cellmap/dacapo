from .raw_gt_dataset import RawGTDataset
from .dataset_config import DatasetConfig
from .arrays import ArrayConfig

import attr


@attr.s
class RawGTDatasetConfig(DatasetConfig):
    """
    This is the standard dataset with a Raw and a GT Array.
    """

    dataset_type = RawGTDataset

    raw_config: ArrayConfig = attr.ib()
    gt_config: ArrayConfig = attr.ib()
