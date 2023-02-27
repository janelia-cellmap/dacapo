from .predictor import Predictor
from dacapo.experiments import Model
from dacapo.experiments.arraytypes import EmbeddingArray
from dacapo.experiments.datasplits.datasets.arrays import NumpyArray
from dacapo.utils.affinities import seg_to_affgraph, padding as aff_padding
from dacapo.utils.balance_weights import balance_weights

from funlib.geometry import Coordinate
from lsd.local_shape_descriptor import LsdExtractor

from scipy import ndimage
import numpy as np
import torch

from typing import List
import time


class AffinitiesPredictor(Predictor):
    def __init__(self, neighborhood: List[Coordinate], lsds: bool = True):
        self.neighborhood = neighborhood
        self.lsds = lsds
        if lsds:
            self._extractor = None
            if self.dims == 2:
                self.num_lsds = 6
            elif self.dims == 3:
                self.num_lsds = 10
            else:
                raise ValueError(
                    f"Cannot compute lsds on volumes with {self.dims} dimensions"
                )
        else:
            self.num_lsds = 0

        self.moving_class_counts = None
        self.moving_lsd_class_counts = None
        self.running_ratio_weight = 0.01

    def extractor(self, voxel_size):
        if self._extractor is None:
            self._extractor = LsdExtractor(self.sigma(voxel_size))

        return self._extractor

    @property
    def dims(self):
        return self.neighborhood[0].dims

    def sigma(self, voxel_size):
        voxel_dist = max(voxel_size)  # arbitrarily chosen
        num_voxels = 10  # arbitrarily chosen
        sigma = voxel_dist * num_voxels
        return Coordinate((sigma,) * self.dims)

    def lsd_pad(self, voxel_size):
        multiplier = 3  # from AddLocalShapeDescriptor Node in funlib.lsd
        padding = Coordinate(self.sigma(voxel_size) * multiplier)
        return padding

    @property
    def num_channels(self):
        return len(self.neighborhood) + self.num_lsds

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
                f"AffinitiesPredictor not implemented for {self.dims} dimensions"
            )

        return Model(architecture, head, eval_activation=torch.nn.Sigmoid())

    def create_target(self, gt):
        # zeros
        assert gt.num_channels is None or gt.num_channels == 1, (
            "Cannot create affinities from ground truth with multiple channels.\n"
            f"GT axes: {gt.axes} with {gt.num_channels} channels"
        )
        label_data = self._grow_boundaries(gt[gt.roi])
        axes = gt.axes
        if gt.num_channels is not None:
            label_data = label_data[0]
        else:
            axes = ["c"] + axes
        t1 = time.time()
        affinities = seg_to_affgraph(label_data, self.neighborhood).astype(np.float32)
        if self.lsds:
            t1 = time.time()
            descriptors = self.extractor(gt.voxel_size).get_descriptors(
                segmentation=label_data,
                voxel_size=gt.voxel_size,
            )
            return NumpyArray.from_np_array(
                np.concatenate([affinities, descriptors], axis=0, dtype=np.float32),
                gt.roi,
                gt.voxel_size,
                axes,
            )
        return NumpyArray.from_np_array(
            affinities,
            gt.roi,
            gt.voxel_size,
            axes,
        )

    def _grow_boundaries(self, gt):

        # get all foreground voxels by erosion of each component
        foreground = np.zeros(shape=gt.shape, dtype=bool)
        masked = None
        for label in np.unique(gt):
            if label == 0:
                continue
            label_mask = gt==label
            # Assume that masked out values are the same as the label we are
            # eroding in this iteration. This ensures that at the boundary to
            # a masked region the value blob is not shrinking.
            if masked is not None:
                label_mask = np.logical_or(label_mask, masked)
            eroded_label_mask = ndimage.binary_erosion(label_mask, iterations=1, border_value=1)
            foreground = np.logical_or(eroded_label_mask, foreground)

        # label new background
        background = np.logical_not(foreground)
        gt[background] = 0
        return gt

    def create_weight(self, gt, target, mask):
        aff_weights, self.moving_class_counts = balance_weights(
            target[target.roi][: self.num_channels - self.num_lsds].astype(np.uint8),
            2,
            slab=tuple(1 if c == "c" else -1 for c in target.axes),
            masks=[mask[target.roi]],
            moving_counts=self.moving_class_counts,
            update_rate=self.running_ratio_weight,
        )
        if self.lsds:
            lsd_weights, self.moving_lsd_class_counts = balance_weights(
                (gt[target.roi] > 0).astype(np.uint8),
                2,
                slab=(-1,) * len(gt.axes),
                masks=[mask[target.roi]],
                moving_counts=self.moving_lsd_class_counts,
                update_rate=self.running_ratio_weight,
            )
            lsd_weights = np.ones(
                (self.num_lsds,) + aff_weights.shape[1:], dtype=aff_weights.dtype
            ) * lsd_weights.reshape((1,) + aff_weights.shape[1:])
            return NumpyArray.from_np_array(
                np.concatenate([aff_weights, lsd_weights], axis=0),
                target.roi,
                target.voxel_size,
                target.axes,
            )
        return NumpyArray.from_np_array(
            aff_weights,
            target.roi,
            target.voxel_size,
            target.axes,
        )

    def gt_region_for_roi(self, target_spec):
        gt_spec = target_spec.copy()
        pad_neg, pad_pos = aff_padding(self.neighborhood, target_spec.voxel_size)
        if self.lsds:
            pad_neg = Coordinate(
                *[
                    max(a, b)
                    for a, b in zip(pad_neg, self.lsd_pad(target_spec.voxel_size))
                ]
            )
            pad_pos = Coordinate(
                *[
                    max(a, b)
                    for a, b in zip(pad_pos, self.lsd_pad(target_spec.voxel_size))
                ]
            )
        gt_spec.roi = gt_spec.roi.grow(pad_neg, pad_pos)
        gt_spec.dtype = None
        return gt_spec

    @property
    def output_array_type(self):
        return EmbeddingArray(self.dims)
