from .predictor import Predictor
from dacapo.experiments import Model
from dacapo.experiments.arraytypes import EmbeddingArray
from dacapo.experiments.datasplits.datasets.arrays import NumpyArray
from dacapo.utils.affinities import seg_to_affgraph, padding as aff_padding
from dacapo.utils.balance_weights import balance_weights
from funlib.geometry import Coordinate
from lsd.train import LsdExtractor
from scipy import ndimage
import numpy as np
import torch
import itertools
from typing import List


class AffinitiesPredictor(Predictor):
    def __init__(
        self,
        neighborhood: List[Coordinate],
        lsds: bool = True,
        num_voxels: int = 20,
        downsample_lsds: int = 1,
        grow_boundary_iterations: int = 0,
        affs_weight_clipmin: float = 0.05,
        affs_weight_clipmax: float = 0.95,
        lsd_weight_clipmin: float = 0.05,
        lsd_weight_clipmax: float = 0.95,
        background_as_object: bool = False,
    ):
        self.neighborhood = neighborhood
        self.lsds = lsds
        self.num_voxels = num_voxels
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
            self.downsample_lsds = downsample_lsds
        else:
            self.num_lsds = 0
        self.grow_boundary_iterations = grow_boundary_iterations
        self.affs_weight_clipmin = affs_weight_clipmin
        self.affs_weight_clipmax = affs_weight_clipmax
        self.lsd_weight_clipmin = lsd_weight_clipmin
        self.lsd_weight_clipmax = lsd_weight_clipmax

        self.background_as_object = background_as_object

    def extractor(self, voxel_size):
        if self._extractor is None:
            self._extractor = LsdExtractor(
                self.sigma(voxel_size), downsample=self.downsample_lsds
            )

        return self._extractor

    @property
    def dims(self):
        return self.neighborhood[0].dims

    def sigma(self, voxel_size):
        voxel_dist = max(voxel_size)  # arbitrarily chosen
        sigma = voxel_dist * self.num_voxels  # arbitrarily chosen
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
        label_data = gt[gt.roi]
        axes = gt.axes
        if gt.num_channels is not None:
            label_data = label_data[0]
        else:
            axes = ["c"] + axes
        affinities = seg_to_affgraph(
            label_data + int(self.background_as_object), self.neighborhood
        ).astype(np.float32)
        if self.lsds:
            descriptors = self.extractor(gt.voxel_size).get_descriptors(
                segmentation=label_data + int(self.background_as_object),
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

    def _grow_boundaries(self, mask, slab):
        # get all foreground voxels by erosion of each component
        foreground = np.zeros(shape=mask.shape, dtype=bool)

        # slab with -1 replaced by shape
        slab = tuple(m if s == -1 else s for m, s in zip(mask.shape, slab))
        slab_ranges = (range(0, m, s) for m, s in zip(mask.shape, slab))

        for ind, start in enumerate(itertools.product(*slab_ranges)):
            slices = tuple(
                slice(start[d], start[d] + slab[d]) for d in range(len(slab))
            )
            mask_slab = mask[slices]
            dilated_mask_slab = ndimage.binary_dilation(
                mask_slab, iterations=self.grow_boundary_iterations
            )
            foreground[slices] = dilated_mask_slab

        # label new background
        background = np.logical_not(foreground)
        mask[background] = 0
        return mask

    def create_weight(self, gt, target, mask, moving_class_counts=None):
        (moving_class_counts, moving_lsd_class_counts) = (
            moving_class_counts if moving_class_counts is not None else (None, None)
        )
        if self.grow_boundary_iterations > 0:
            mask_data = self._grow_boundaries(
                mask[target.roi], slab=tuple(1 if c == "c" else -1 for c in target.axes)
            )
        else:
            mask_data = mask[target.roi]
        aff_weights, moving_class_counts = balance_weights(
            target[target.roi][: self.num_channels - self.num_lsds].astype(np.uint8),
            2,
            slab=tuple(1 if c == "c" else -1 for c in target.axes),
            masks=[mask_data],
            moving_counts=moving_class_counts,
            clipmin=self.affs_weight_clipmin,
            clipmax=self.affs_weight_clipmax,
        )
        if self.lsds:
            lsd_weights, moving_lsd_class_counts = balance_weights(
                (gt[target.roi] > 0).astype(np.uint8),
                2,
                slab=(-1,) * len(gt.axes),
                masks=[mask_data],
                moving_counts=moving_lsd_class_counts,
                clipmin=self.lsd_weight_clipmin,
                clipmax=self.lsd_weight_clipmax,
            )
            lsd_weights = np.ones(
                (self.num_lsds,) + aff_weights.shape[1:], dtype=aff_weights.dtype
            ) * lsd_weights.reshape((1,) + aff_weights.shape[1:])
            return NumpyArray.from_np_array(
                np.concatenate([aff_weights, lsd_weights], axis=0),
                target.roi,
                target.voxel_size,
                target.axes,
            ), (moving_class_counts, moving_lsd_class_counts)
        return NumpyArray.from_np_array(
            aff_weights,
            target.roi,
            target.voxel_size,
            target.axes,
        ), (moving_class_counts, moving_lsd_class_counts)

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
        gt_spec.roi = gt_spec.roi.grow(pad_neg, pad_pos).snap_to_grid(
            target_spec.voxel_size
        )
        gt_spec.dtype = None
        return gt_spec

    @property
    def output_array_type(self):
        return EmbeddingArray(self.dims)
