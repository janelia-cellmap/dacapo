"""
This script is for configuration setup of data splits for training and validation in funkelab daCapo python library. 
It includes importing necessary modules, defining the TrainValidateDataSplitConfig class  and setting configurations setups.

Imports:
    TrainValidateDataSplit: A class to split data for training and validating.
    DataSplitConfig: A configuration setup for data splitting.
    DatasetConfig: A configuration setup for dataset.
    attr: An attribute handling library in python.
    List: A built-in Python function - data type that holds an ordered collection of items.

Class:
    TrainValidateDataSplitConfig(DataSplitConfig: A class that inherits from `DataSplitConfig`.
        This is the standard configuration set up for Train/Validate DataSplit in daCapo Python Library.

Attributes:
    datasplit_type: The type of datasplit to be used, which is TrainValidateDataSplit. 
    train_configs: A list of all the configurations for the datasets used for training. 
        metadata {'help_text': Explains where to use it - "All of the datasets to use for training."}
    validate_configs: A list of all the configurations for the datasets used for validation.
        metadata {'help_text': Explains where to use it - "All of the datasets to use for validation."}
"""