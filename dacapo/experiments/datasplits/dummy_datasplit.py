from .datasplit import DataSplit
from .datasets import Dataset

from typing import List


class DummyDataSplit(DataSplit):
    train: List[Dataset]
    validate: List[Dataset]

    def __init__(self, datasplit_config):
        super().__init__()

        self.train = [
            datasplit_config.train_config.dataset_type(datasplit_config.train_config)
        ]
        self.validate = []
