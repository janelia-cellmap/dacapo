"""
DummyDataSplit is a class derived from the DataSplit class which is used to setup a simple list of one dataset for training purposes. 
Validation datasets are left as an empty list in this class.

Attributes:
----------
train : list
    List containing the training dataset(s).
validate : list
    An empty list for validation data. It does not contain any validation dataset in this class.

Methods:
----------
__init__(self, datasplit_config):
    Initializes the DummyDataSplit instance with the configuration setup for training.
"""

from .datasplit import DataSplit
from .datasets import Dataset

from typing import List


class DummyDataSplit(DataSplit):
    """A class for creating a simple train dataset and no validation dataset.

    It is derived from `DataSplit` class.

    ...
    Attributes
    ----------
    train : list
        The list containing training datasets. In this class, it contains only one dataset for training.
    validate : list
        The list containing validation datasets. In this class, it is an empty list as no validation dataset is set.

    Methods
    -------
    __init__(self, datasplit_config):
        The constructor for DummyDataSplit class. It initialises a list with training datasets according to the input configuration.
    """
    train: List[Dataset]
    validate: List[Dataset]

    def __init__(self, datasplit_config):
        """Constructor method for initializing the instance of `DummyDataSplit` class. It sets up the list of training datasets based on the passed configuration.

        Parameters
        ----------
        datasplit_config : DatasplitConfig
            The configuration setup for processing the datasets into the training sets.
        """
        super().__init__()

        self.train = [
            datasplit_config.train_config.dataset_type(datasplit_config.train_config)
        ]
        self.validate = []