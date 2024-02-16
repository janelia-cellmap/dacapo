class RawGTDataset(Dataset):
    """
    A class to represent a raw ground truth dataset.

    Attributes:
        raw (Array): The raw data array.
        gt (Array): The ground truth data array.
        mask (Optional[Array]): Optional mask for the data. Defaults to None. 
        sample_points (Optional[List[Coordinate]]): Optional list of coordinates. Defaults to None.
        
    Args:
        dataset_config (object): The configuration information for the dataset.

    """

    raw: Array
    gt: Array
    mask: Optional[Array]
    sample_points: Optional[List[Coordinate]]

    def __init__(self, dataset_config):
        """
        Construct all the necessary attributes for the RawGTDataset object.

        Args:
            dataset_config (object): The configuration information for the dataset.

        """

        self.name = dataset_config.name
        self.raw = dataset_config.raw_config.array_type(dataset_config.raw_config)
        self.gt = dataset_config.gt_config.array_type(dataset_config.gt_config)
        self.mask = (
            dataset_config.mask_config.array_type(dataset_config.mask_config)
            if dataset_config.mask_config is not None
            else None
        )
        self.sample_points = dataset_config.sample_points
        self.weight = dataset_config.weight