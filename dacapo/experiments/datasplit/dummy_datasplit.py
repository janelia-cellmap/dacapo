from .datasplit import DataSplit


class DummyDataSplit(DataSplit):

    def __init__(self, datasplit_config):

        super().__init__()

        self.train = datasplit_config.train_config.dataset_type(datasplit_config.train_config)
        self.validate = datasplit_config.validate_config.dataset_type(datasplit_config.validate_config)

