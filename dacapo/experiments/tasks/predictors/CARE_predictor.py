from .predictor import Predictor
from dacapo.experiments import Model
from dacapo.experiments.arraytypes import IntensitiesArray
from dacapo.experiments.datasplits.datasets.arrays import NumpyArray, ZarrArray

from funlib.geometry import Coordinate

import numpy as np
import torch


class CAREPredictor(Predictor):
    def __init__(self, num_channels, dims):
        self.num_channels = num_channels
        self.dims = dims

    def create_model(self, architecture):
        if self.dims == 2:
            head = torch.nn.Conv2d(
                architecture.num_out_channels, self.num_channels, kernel_size=1
            )
        elif self.dims == 3:
            head = torch.nn.Conv3d(
                architecture.num_out_channels, self.num_channels, kernel_size=1
            )
        else:
            raise NotImplementedError(
                f"CAREPredictor not implemented for {self.dims} dimensions"
            )

        return Model(architecture, head)

    def create_target(self, gt):
        return gt

    def create_weight(self, gt, target=None, mask=None):
        if mask is None:
            # array of ones
            return NumpyArray.from_np_array(
                np.ones(gt.data.shape),
                gt.roi,
                gt.voxel_size,
                gt.axes,
            )
        else:
            return mask

    @property
    def output_array_type(self):
        return IntensitiesArray({"channels": {n: str(n) for n in range(self.num_channels)}}, min=0., max=1.)


