from .raw_gt_dataset import RawGTDataset
from .dataset_config import DatasetConfig
from .arrays import ArrayConfig

from funlib.geometry import Coordinate

import attr

from typing import Optional, List


@attr.s
class RawGTDatasetConfig(DatasetConfig):
    """
    This is a configuration class for the standard dataset with both raw and GT Array.

    The configuration includes array configurations for raw data, ground truth data and mask data.
    The configuration for ground truth (GT) data is mandatory, whereas configurations for raw
    and mask data are optional. It also includes an optional list of points around which training samples
    will be extracted.

    Attributes:
        dataset_type (class): The type of dataset that is being configured.
        raw_config (Optional[ArrayConfig]): Configuration for the raw data associated with this dataset.
        gt_config (Optional[ArrayConfig]): Configuration for the ground truth data associated with this dataset.
        mask_config (Optional[ArrayConfig]): An optional mask configuration that sets the loss
                                             equal to zero on voxels where the mask is 1.
        sample_points (Optional[List[Coordinate]]): An optional list of points around which
                                                    training samples will be extracted.
    Methods:
        verify: A method to verify the validity of the configuration.
    Notes:
        This class is used to create a configuration object for the standard dataset with both raw and GT Array.
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
