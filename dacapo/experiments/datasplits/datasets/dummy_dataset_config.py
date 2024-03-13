from .dummy_dataset import DummyDataset
from .dataset_config import DatasetConfig
from .arrays import ArrayConfig, DummyArrayConfig

import attr

from typing import Tuple


@attr.s
class DummyDatasetConfig(DatasetConfig):
    """
    A dummy configuration class for test datasets.

    Attributes:
        dataset_type : Clearly mentions the type of dataset
        raw_config : This attribute holds the configurations related to dataset arrays.

    Methods:
        verify: A dummy verification method for testing purposes, always returns False and a message.
    """

    dataset_type = DummyDataset

    raw_config: ArrayConfig = attr.ib(DummyArrayConfig(name="dummy_array"))

    def verify(self) -> Tuple[bool, str]:
        """A dummy method that always indicates the dataset config is not valid.

        Returns:
            A tuple of False and a message indicating the invalidity.
        """

        return False, "This is a DummyDatasetConfig and is never valid"
