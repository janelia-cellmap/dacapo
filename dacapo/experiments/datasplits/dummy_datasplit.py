from .datasplit import DataSplit
from .datasets import Dataset

from typing import List


class DummyDataSplit(DataSplit):
    """
    A class for creating a simple train dataset and no validation dataset. It is derived from `DataSplit` class.
    It is used to split the data into training and validation datasets. The training and validation datasets are
    used to train and validate the model respectively.

    Attributes:
        train : list
            The list containing training datasets. In this class, it contains only one dataset for training.
        validate : list
            The list containing validation datasets. In this class, it is an empty list as no validation dataset is set.
    Methods:
        __init__(self, datasplit_config):
            The constructor for DummyDataSplit class. It initialises a list with training datasets according to the input configuration.
    Notes:
        This class is used to split the data into training and validation datasets.
    """

    train: List[Dataset]
    validate: List[Dataset]

    def __init__(self, datasplit_config):
        """
        Constructor method for initializing the instance of `DummyDataSplit` class. It sets up the list of training datasets based on the passed configuration.

        Args:
            datasplit_config : obj
                The configuration to initialize the DummyDataSplit class.
        Raises:
            Exception
                If the model setup cannot be loaded, an Exception is thrown which is logged and handled by training the model without head matching.
        Examples:
            >>> dummy_datasplit = DummyDataSplit(datasplit_config)
        Notes:
            This function is called by the DummyDataSplit class to initialize the DummyDataSplit class with specified config to split the data into training and validation datasets.
        """
        super().__init__()

        self.train = [
            datasplit_config.train_config.dataset_type(datasplit_config.train_config)
        ]
        self.validate = []
