from .dataset import Dataset
from .arrays import Array

from funlib.geometry import Coordinate

from typing import Optional, List


class RawGTDataset(Dataset):
    """
    A dataset that contains raw and ground truth data. Optionally, it can also contain a mask.

    Attributes:
        raw: Array
            The raw data.
        gt: Array
            The ground truth data.
        mask: Optional[Array]
            The mask data.
        sample_points: Optional[List[Coordinate]]
            The sample points in the graph.
        weight: Optional[float]
            The weight of the dataset.
    Methods:
        __init__(dataset_config):
            Initialize the dataset.
    Notes:
        This class is a base class and should not be instantiated.
    """

    raw: Array
    gt: Array
    mask: Optional[Array]
    sample_points: Optional[List[Coordinate]]

    def __init__(self, dataset_config):
        """
        Initialize the dataset.

        Args:
            dataset_config: DataSplitConfig
                The configuration of the dataset.
        Raises:
            NotImplementedError
                If the method is not implemented in the derived class.
        Examples:
            >>> dataset = RawGTDataset(dataset_config)
        Notes:
            This method is used to initialize the dataset.
        """
        self.name = dataset_config.name
        try:
            self.raw = dataset_config.raw_config.array_type(dataset_config.raw_config)
            self.gt = dataset_config.gt_config.array_type(dataset_config.gt_config)
            self.mask = (
                dataset_config.mask_config.array_type(dataset_config.mask_config)
                if dataset_config.mask_config is not None
                else None
            )
        except Exception as e:
            raise Exception(
                f"Error loading arrays for dataset {self.name}: {e} \n {dataset_config}"
            )
        self.sample_points = dataset_config.sample_points
        self.weight = dataset_config.weight
