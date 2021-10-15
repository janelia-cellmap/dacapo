from .dataset import Dataset
from .arraystores import ArrayStore

class DummyDataset(Dataset):

    raw: ArrayStore = None

    def __init__(self, dataset_config):

        super().__init__()

        self.raw = dataset_config.raw_config.array_store_type(dataset_config.raw_config)

