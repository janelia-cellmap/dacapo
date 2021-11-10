from .datasplit import DataSplit
from .datasets import Dataset


class TrainValidateDataSplit(DataSplit):

    train: Dataset = None
    validate: Dataset = None

    def __init__(self, datasplit_config):

        super().__init__()

        self.train = [
            train_config.dataset_type(train_config)
            for train_config in datasplit_config.train_configs
        ]
        self.validate = [
            validate_config.dataset_type(validate_config)
            for validate_config in datasplit_config.validate_configs
        ]
