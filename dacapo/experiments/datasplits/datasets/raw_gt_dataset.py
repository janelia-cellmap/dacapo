from .dataset import Dataset
from .arrays import Array

from funlib.geometry import Coordinate

from typing import Optional, List


class RawGTDataset(Dataset):
    raw: Array
    gt: Array
    mask: Optional[Array]
    sample_points: Optional[List[Coordinate]]

    def __init__(self, dataset_config):
        self.name = dataset_config.name
        self.raw = dataset_config.raw_config.array_type(dataset_config.raw_config)
        self.gt = dataset_config.gt_config.array_type(dataset_config.gt_config)
        self.mask = (
            dataset_config.mask_config.array_type(dataset_config.mask_config)
            if dataset_config.mask_config is not None
            else None
        )
        self.sample_points = dataset_config.sample_points
        self.weight = dataset_config.weight
