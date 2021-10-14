from .dataset import Dataset


class DummyDataset(Dataset):

    def __init__(self, dataset_config):

        super().__init__()

        self.raw = dataset_config.raw_config.dataset_type(dataset_config.raw_config)

