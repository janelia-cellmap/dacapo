from .predictor import Predictor
from dacapo.experiments import Model
from dacapo.experiments.arraytypes import ProbabilityArray
from dacapo.experiments.datasplits.datasets.arrays import NumpyArray

import numpy as np
import torch

from typing import List
import logging

logger = logging.getLogger(__name__)


class OneHotPredictor(Predictor):
    def __init__(self, classes: List[str]):
        self.classes = classes

    @property
    def embedding_dims(self):
        return len(self.classes)

    def create_model(self, architecture):

        head = torch.nn.Conv3d(
            architecture.num_out_channels, self.embedding_dims, kernel_size=3
        )

        return Model(architecture, head)

    def create_target(self, gt):
        one_hots = self.process(gt.data)
        return NumpyArray.from_np_array(
            one_hots,
            gt.roi,
            gt.voxel_size,
            gt.axes,
        )

    def create_weight(self, gt, target, mask, moving_class_counts=None):
        return (
            NumpyArray.from_np_array(
                np.ones(target.data.shape),
                target.roi,
                target.voxel_size,
                target.axes,
            ),
            None,
        )

    @property
    def output_array_type(self):
        return ProbabilityArray(self.classes)

    def process(
        self,
        labels: np.ndarray,
    ):
        # TODO: Assumes labels has a singleton channel dim and channel dim is first
        one_hots = np.zeros((self.embedding_dims,) + labels.shape[1:], dtype=np.uint8)
        for i, _ in enumerate(self.classes):
            one_hots[i] += labels[0] == i
        return one_hots
