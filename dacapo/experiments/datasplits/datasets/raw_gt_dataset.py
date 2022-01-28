from .dataset import Dataset
from .arrays import Array

from typing import Optional

class RawGTDataset(Dataset):

    raw: Array = None
    gt: Array = None
    mask: Optional[Array] = None

    def __init__(self, dataset_config):

        self.name = dataset_config.name
        self.raw = dataset_config.raw_config.array_type(dataset_config.raw_config)
        self.gt = dataset_config.gt_config.array_type(dataset_config.gt_config)
        self.mask = (
            dataset_config.mask_config.array_type(dataset_config.mask_config)
            if dataset_config.mask_config is not None
            else None
        )
