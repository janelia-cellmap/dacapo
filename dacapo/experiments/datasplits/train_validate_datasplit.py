"""
This script is a part of Funkelab DaCapo Python library and creates a class to implement training and validate data splits, wherein,
DataSplit is inherited and the class TrainValidateDataSplit extends it with train and validate list. It also comprises a function to
initialize the data split configurations and assign the respective dataset types.

Classes:
-------
`TrainValidateDataSplit (DataSplit)`
    Implements a data-split for train and validate data sets.

Functions:
---------
`__init__(self, datasplit_config)`
    Initializes the datasplit_config for train and validate data.

"""


class TrainValidateDataSplit(DataSplit):
    """
    Represents a class that divides data into training and testing datasets. Inherits from DataSplit class.
 
    Attributes:
    ----------
    `train (List[Dataset])`: A list of training datasets.
    `validate (List[Dataset])`: A list of validation datasets.
    """
    train: List[Dataset]
    validate: List[Dataset]

    def __init__(self, datasplit_config):
        """
        Initializes the TrainValidateDataSplit with the given configuration.

        The constructor splits the `datasplit_config` into different configurations and extracts respective dataset type for each
        configuration.

        Parameters:
        ----------
        `datasplit_config`: A data split configuration object.
        """
        super().__init__()

        self.train = [
            train_config.dataset_type(train_config)
            for train_config in datasplit_config.train_configs
        ]
        self.validate = [
            validate_config.dataset_type(validate_config)
            for validate_config in datasplit_config.validate_configs
        ]