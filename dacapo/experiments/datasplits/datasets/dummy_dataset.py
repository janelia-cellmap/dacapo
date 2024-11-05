from .dataset import Dataset
from funlib.persistence import Array


class DummyDataset(Dataset):
    """
    DummyDataset is a child class of the Dataset. This class has property 'raw' of Array type and a name.

    Attributes:
        raw: Array
            The raw data.
    Methods:
        __init__(dataset_config):
            Initializes the array type 'raw' and name for the DummyDataset instance.
    Notes:
        This class is used to create a dataset with raw data.
    """

    raw: Array

    def __init__(self, dataset_config):
        """
        Initializes the array type 'raw' and name for the DummyDataset instance.

        Args:
            dataset_config (object): an instance of a configuration class that includes the name and
            raw configuration of the data.
        Raises:
            NotImplementedError
                If the method is not implemented in the derived class.
        Examples:
            >>> dataset = DummyDataset(dataset_config)
        Notes:
            This method is used to initialize the dataset.
        """
        super().__init__()
        self.name = dataset_config.name
        self.raw = dataset_config.raw_config.array()
