"""
dacapo package

This package provides the core functionalities for managing different datasets. It includes definitions for Dataset,
DatasetConfig, DummyDataset, DummyDatasetConfig, RawGTDataset, RawGTDatasetConfig. These classes allow convenient and 
manageable handling of large datasets.
  
Modules
-------
.dataset : Base 'Dataset' definition. It is the building block for other classes.
.dataset_config : A configuration script for datasets.
.dummy_dataset : A dummy dataset for testing purposes.
.dummy_dataset_config : Configuration settings for the dummy dataset.
.raw_gt_dataset : A dataset class for handling raw ground-truth datasets.
.raw_gt_dataset_config : Configuration for the raw ground-truth dataset class.

Each module has its own functionality provided to assist with the handling of large datasets.
"""

from .dataset import Dataset
from .dataset_config import DatasetConfig
from .dummy_dataset import DummyDataset
from .dummy_dataset_config import DummyDatasetConfig
from .raw_gt_dataset import RawGTDataset
from .raw_gt_dataset_config import RawGTDatasetConfig