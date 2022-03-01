from .predictor import Predictor
from dacapo.experiments import Model
from dacapo.experiments.arraytypes import EmbeddingArray
from dacapo.experiments.datasplits.datasets.arrays import NumpyArray

import numpy as np
import torch


class DummyPredictor(Predictor):
    def __init__(self, embedding_dims):
        self.embedding_dims = embedding_dims

    def create_model(self, architecture):

        head = torch.nn.Conv3d(
            architecture.num_out_channels, self.embedding_dims, kernel_size=3
        )

        return Model(architecture, head)

    def create_target(self, gt):
        # zeros
        return NumpyArray.from_np_array(
            np.zeros((self.embedding_dims,) + gt.data.shape[-gt.dims:]),
            gt.roi,
            gt.voxel_size,
            ["c"] + gt.axes,
        )

    def create_weight(self, gt, target, mask):
        return NumpyArray.from_np_array(
            np.ones(target.data.shape),
            target.roi,
            target.voxel_size,
            target.axes,
        )

    @property
    def output_array_type(self):
        return EmbeddingArray(self.embedding_dims)
