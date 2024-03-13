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
