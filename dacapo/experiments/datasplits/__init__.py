"""
Module containing all the necessary classes and configurations for effective data splitting.
The data splitting approach is determined by the application and dataset requirements.

The module includes classes for data splitting, data split configuration, dummy data split, 
dummy data split configuration, train validate data split and its configuration.

Classes:
    DataSplit: Class for splitting data based on a given config.
    DataSplitConfig: Configuration class for controlling the data split.
    DummyDataSplit: Class for creating a dummy data split based on a given config.
    DummyDataSplitConfig: Configuration class for controlling the dummy data split.
    TrainValidateDataSplit: Class for creating a training and validation data split.
    TrainValidateDataSplitConfig: Configuration class for controlling the training 
    and validation data split.

Imports:
    datasplit: Provides the main data splitting class.
    datasplit_config: Provides the data splitting configuration class.
    dummy_datasplit: Provides the class for dummy data splitting.
    dummy_datasplit_config: Provides the dummy data splitting configuration class.
    train_validate_datasplit: Provides the class for train and validation data splitting.
    train_validate_datasplit_config: Provides the train and validation data splitting 
    configuration class.
"""

from .datasplit import DataSplit
from .datasplit_config import DataSplitConfig
from .dummy_datasplit import DummyDataSplit
from .dummy_datasplit_config import DummyDataSplitConfig
from .train_validate_datasplit import TrainValidateDataSplit
from .train_validate_datasplit_config import TrainValidateDataSplitConfig
