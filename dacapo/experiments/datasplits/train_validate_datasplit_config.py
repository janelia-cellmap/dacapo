from .train_validate_datasplit import TrainValidateDataSplit
from .datasplit_config import DataSplitConfig
from .datasets import DatasetConfig

import attr

from typing import List


@attr.s
class TrainValidateDataSplitConfig(DataSplitConfig):
    """
    This is the standard Train/Validate DataSplit config.
    """

    datasplit_type = TrainValidateDataSplit

    train_configs: List[DatasetConfig] = attr.ib(
        metadata={"help_text": "All of the datasets to use for training"}
    )
    validate_configs: List[DatasetConfig] = attr.ib(
        metadata={
            "help_text": "All of the datasets to use for validation. Currently only supports 1 volume!"
        }
    )
