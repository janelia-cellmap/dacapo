from .dataset import Dataset
from .arrays import Array


class RawGTDataset(Dataset):

    raw: Array = None
    gt: Array = None

    def __init__(self, dataset_config):

        self.raw = dataset_config.raw_config.array_type(dataset_config.raw_config)
        self.gt = dataset_config.gt_config.array_type(dataset_config.gt_config)
