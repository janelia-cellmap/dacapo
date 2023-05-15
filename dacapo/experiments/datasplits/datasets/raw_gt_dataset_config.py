from .raw_gt_dataset import RawGTDataset
from .dataset_config import DatasetConfig
from .arrays import ArrayConfig

from funlib.geometry import Coordinate

import attr

from typing import Optional, List


@attr.s
class RawGTDatasetConfig(DatasetConfig):
    """
    This is the standard dataset with a Raw and a GT Array.
    """

    dataset_type = RawGTDataset

    raw_config: Optional[ArrayConfig] = attr.ib(
        default=None,
        metadata={"help_text": "Config for the raw data associated with this dataset."},
    )
    gt_config: Optional[ArrayConfig] = attr.ib(
        default=None,
        metadata={
            "help_text": "Config for the ground truth data associated with this dataset."
        },
    )
    mask_config: Optional[ArrayConfig] = attr.ib(
        default=None,
        metadata={
            "help_text": "An optional mask that sets the loss equal to zero on voxels where "
            "the mask is 1"
        },
    )
    sample_points: Optional[List[Coordinate]] = attr.ib(
        default=None,
        metadata={
            "help_text": "An optional list of points around which training samples will be "
            "extracted."
        },
    )
