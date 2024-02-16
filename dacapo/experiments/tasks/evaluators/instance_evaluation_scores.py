class DacapoDataModule(pl.LightningDataModule):
    """
    DacapoDataModule is a PyTorch LightningDataModule that is responsible for the process of loading, 
    processing, and preparing datasets for model training and evaluation.

    Attributes:
        dataset_name (str): Name of the dataset.
        batch_size (int): Batch size for data sequencing.
        eval_batch_size (int): Batch size specific for evaluation.
        num_workers (int): Number of workers to utilize in dataloading process.
        split: Indices for splitting the dataset.
        normalize (bool): Flag indicating whether dataset normalization should be applied.
        split_method (str): Method for splitting the datasets: 'seg', 'equally'.
        seed (int): Seed value for reproducibility.
    """

    def __init__(self, dataset_name,
                 batch_size=1,
                 eval_batch_size=1,
                 normalize=False,
                 num_workers=1,
                 split=(0, 700, 840, 840),
                 split_method='seg',
                 seed=1234,
                 ):
        super().__init__()

    def setup(self, stage):
        """
        Function that handles the main data loading and dataset splitting tasks.
        
        Args:
            stage (str): The current stage ('fit' or 'test') for Datamodule.
        """
        if stage == 'fit' or stage is None:

    def train_dataloader(self):
        """
        Loads and returns the training dataloader.
        
        Returns:
            dataloader for training data.
        """

    def val_dataloader(self):
        """
        Loads and returns the validation dataloader.

        Returns:
            dataloader for validation data.
        """

    def test_dataloader(self):
        """
        Loads and returns the test dataloader.

        Returns:
            dataloader for test data.
        """
