from dacapo.experiments.tasks import TaskConfig
from pathlib import Path
from typing import List
from enum import Enum
import os
import numpy as np
from dacapo.experiments.datasplits.datasets.arrays import (
    ArrayConfig,
    ZarrArrayConfig,
    ZarrArray,
    ResampledArrayConfig,
    BinarizeArrayConfig,
    IntensitiesArrayConfig,
    IntensitiesArray,
    MissingAnnotationsMaskConfig,
    OnesArrayConfig,
    ConcatArrayConfig,
    LogicalOrArrayConfig,
    CropArrayConfig,
    MergeInstancesArrayConfig,
)
import logging

logger = logging.getLogger(__name__)


class DatasetSpec:
    raw_path: Path
    raw_dataset: str
    # raw_resolution: List[int]
    gt_path: Path
    gt_dataset: str
    # gt_resolution: List[int]
    mask_path: Path
    mask_dataset: str
    sample_points: Path  # to numpy array of coordinates

    def __init__(
        self,
        raw_path: Path,
        raw_dataset: str,
        gt_path: Path,
        gt_dataset: str,
        mask_path: Optional[Path],
        mask_dataset: Optional[str],
        sample_points: Optional[Path],
    ):
        self.raw_path = raw_path
        self.raw_dataset = raw_dataset
        self.gt_path = gt_path
        self.gt_dataset = gt_dataset
        self.mask_path = mask_path
        self.mask_dataset = mask_dataset
        self.sample_points = sample_points
