from .predictor import Predictor
from dacapo.experiments import Model
from dacapo.experiments.datasplits.datasets.arrays import NumpyArray, ZarrArray

from funlib.geometry import Coordinate  # TODO: pip install

import numpy as np
import torch


class CycleGANPredictor(Predictor):
    def __init__(self, num_channels):
        self.num_channels = num_channels

    def create_model(self, netG1, netG2):
        if self.dims == 2:
            netG1 = torch.nn.Conv2d(
                netG1.num_out_channels, self.num_channels, kernel_size=1
            )
            netG2 = torch.nn.Conv2d(
                netG2.num_out_channels, self.num_channels, kernel_size=1
            )
        elif self.dims == 3:
            netG1 = torch.nn.Conv3d(
                netG1.num_out_channels, self.num_channels, kernel_size=1
            )
            netG2 = torch.nn.Conv3d(
                netG2.num_out_channels, self.num_channels, kernel_size=1
            )
        else:
            raise NotImplementedError(
                f"CycleGANPredictor not implemented for {self.dims} dimensions"
            )
        # TODO:
        # return Model(architecture, head)
        pass

    def create_target(self, gt):
        return gt

    def create_weight(self, gt):
        # ones
        return NumpyArray.from_np_array(
            np.ones(gt.data.shape),
            gt.roi,
            gt.voxel_size,
            gt.axes,
        )

    @property
    def output_array_type(self):
        return ZarrArray(self.num_channels)

    def gt_region_for_roi(self, target_spec):
        if self.mask_distances:
            gt_spec = target_spec.copy()
            gt_spec.roi = gt_spec.roi.grow(
                Coordinate((self.max_distance,) * gt_spec.voxel_size.dims),
                Coordinate((self.max_distance,) * gt_spec.voxel_size.dims),
            ).snap_to_grid(gt_spec.voxel_size, mode="shrink")
        else:
            gt_spec = target_spec.copy()
        return gt_spec

    def padding(self, gt_voxel_size: Coordinate) -> Coordinate:
        return Coordinate((self.max_distance,) * gt_voxel_size.dims)
