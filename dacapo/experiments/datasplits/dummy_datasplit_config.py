from .dummy_datasplit import DummyDataSplit
from .datasplit_config import DataSplitConfig
from .datasets import DatasetConfig, DummyDatasetConfig

import attr

from typing import Tuple


@attr.s
class DummyDataSplitConfig(DataSplitConfig):
    """
    A simple class representing config for Dummy DataSplit.

    This class is derived from 'DataSplitConfig' and is initialized with
    'DatasetConfig' for training dataset.

    Attributes:
        datasplit_type: Class of dummy data split functionality.
        train_config: Config for the training dataset. Defaults to DummyDatasetConfig.
    Methods:
        verify()
            A method for verification. This method always return 'False' plus
            a string indicating the condition.
    Notes:
        This class is used to represent the configuration for Dummy DataSplit.

    """

    # Members with default values
    datasplit_type = DummyDataSplit
    train_config: DatasetConfig = attr.ib(DummyDatasetConfig(name="dummy_dataset"))

    def verify(self) -> Tuple[bool, str]:
        """
        A method for verification. This method always return 'False' plus
        a string indicating the condition.

        Returns:
            Tuple[bool, str]: A tuple contains a boolean 'False' and a string.
        Examples:
            >>> dummy_datasplit_config = DummyDataSplitConfig(train_config)
            >>> dummy_datasplit_config.verify()
            (False, "This is a DummyDataSplit and is never valid")
        Notes:
            This method is used to verify the configuration of DummyDataSplit.
        """
        return False, "This is a DummyDataSplit and is never valid"
