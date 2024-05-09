from .train_validate_datasplit import TrainValidateDataSplit
from .datasplit_config import DataSplitConfig
from .datasets import DatasetConfig

import attr

from typing import List


@attr.s
class TrainValidateDataSplitConfig(DataSplitConfig):
    """
    This is the standard Train/Validate DataSplit config. It contains a list of
    training and validation datasets. This class is used to split the data into
    training and validation datasets. The training and validation datasets are
    used to train and validate the model respectively.

    Attributes:
        train_configs : list
            The list of training datasets.
        validate_configs : list
            The list of validation datasets.
    Methods:
        __init__(datasplit_config)
            Initializes the TrainValidateDataSplitConfig class with specified config to
            split the data into training and validation datasets.
    Notes:
        This class is used to split the data into training and validation datasets.

    """

    datasplit_type = TrainValidateDataSplit

    train_configs: List[DatasetConfig] = attr.ib(
        metadata={"help_text": "All of the datasets to use for training."}
    )
    validate_configs: List[DatasetConfig] = attr.ib(
        metadata={"help_text": "All of the datasets to use for validation."}
    )
