from .dataset import Dataset
from .arrays import Array


class DummyDataset(Dataset):

    raw: Array = None

    def __init__(self, dataset_config):

        super().__init__()
        self.name = dataset_config.name
        self.raw = dataset_config.raw_config.array_type(dataset_config.raw_config)
