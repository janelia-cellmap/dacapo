from .datasplit import DataSplit
from .datasets import Dataset


class DummyDataSplit(DataSplit):

    train: Dataset = None

    def __init__(self, datasplit_config):

        super().__init__()

        self.train = [
            datasplit_config.train_config.datasplit_type(datasplit_config.train_config)
        ]
