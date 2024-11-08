from .predictor import Predictor
from dacapo.experiments import Model
from dacapo.experiments.arraytypes import EmbeddingArray
from dacapo.tmp import np_to_funlib_array
from dacapo.utils.affinities import seg_to_affgraph, padding as aff_padding
from dacapo.utils.balance_weights import balance_weights
from dacapo.tmp import np_to_funlib_array
from funlib.geometry import Coordinate
from funlib.persistence import Array
from lsd.train import LsdExtractor
from scipy import ndimage
import numpy as np
import torch
import itertools
from typing import List


class AffinitiesPredictor(Predictor):
    """
    A predictor for generating affinity predictions from input data.

    Attributes:
        neighborhood (List[Coordinate]): The neighborhood.
        lsds (bool): Whether to compute local shape descriptors.
        num_voxels (int): The number of voxels.
        downsample_lsds (int): The downsample rate for LSDs.
        grow_boundary_iterations (int): The number of iterations to grow the boundary.
        affs_weight_clipmin (float): The minimum weight for affinities.
        affs_weight_clipmax (float): The maximum weight for affinities.
        lsd_weight_clipmin (float): The minimum weight for LSDs.
        lsd_weight_clipmax (float): The maximum weight for LSDs.
        background_as_object (bool): Whether to treat the background as an object.
    Methods:
        __init__(
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
            background_as_object: bool = False
        ): Initializes the AffinitiesPredictor.
        extractor(self, voxel_size): Get the LSD extractor.
        dims: Get the number of dimensions.
        sigma(self, voxel_size): Compute the sigma value for LSD computation.
        lsd_pad(self, voxel_size): Compute the padding for LSD computation.
        num_channels: Get the number of channels.
        create_model(self, architecture): Create the model.
        create_target(self, gt): Create the target data.
        _grow_boundaries(self, mask, slab): Grow the boundaries of the mask.
        create_weight(self, gt, target, mask, moving_class_counts=None): Create the weight data.
        gt_region_for_roi(self, target_spec): Get the ground truth region for the target region of interest (ROI).
        output_array_type: Get the output array type.
    Notes:
        This is a subclass of Predictor.
    """

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
        """
        Initializes the AffinitiesPredictor.

        Args:
            neighborhood (List[Coordinate]): The neighborhood.
        Raises:
            ValueError: If the number of dimensions is not 2 or 3.
        Examples:
            >>> neighborhood = [Coordinate((0, 1)), Coordinate((1, 0))]
        """
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
        """
        Get the LSD extractor.

        Args:
            voxel_size (Coordinate): The voxel size.
        Returns:
            LsdExtractor: The LSD extractor.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> extractor = predictor.extractor(voxel_size)
        """
        if self._extractor is None:
            self._extractor = LsdExtractor(
                self.sigma(voxel_size), downsample=self.downsample_lsds
            )

        return self._extractor

    @property
    def dims(self):
        """
        Get the number of dimensions.

        Returns:
            int: The number of dimensions.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> predictor.dims

        """
        return self.neighborhood[0].dims

    def sigma(self, voxel_size):
        """
        Compute the sigma value for LSD computation.

        Args:
            voxel_size (Coordinate): The voxel size.
        Returns:
            Coordinate: The sigma value.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> predictor.sigma(voxel_size)
        """
        voxel_dist = max(voxel_size)  # arbitrarily chosen
        sigma = voxel_dist * self.num_voxels  # arbitrarily chosen
        return Coordinate((sigma,) * self.dims)

    def lsd_pad(self, voxel_size):
        """
        Compute the padding for LSD computation.

        Args:
            voxel_size (Coordinate): The voxel size.
        Returns:
            Coordinate: The padding value.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> predictor.lsd_pad(voxel_size)
        """
        multiplier = 3  # from AddLocalShapeDescriptor Node in funlib.lsd
        padding = Coordinate(self.sigma(voxel_size) * multiplier)
        return padding

    def create_model(self, architecture):
        """
        Create the model.

        Args:
            architecture: The architecture for the model.
        Returns:
            Model: The created model.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> model = predictor.create_model(architecture)
        """
        if self.dims == 2:
            head = torch.nn.Conv2d(
                architecture.num_out_channels, len(self.neighborhood), kernel_size=1
            )
        elif self.dims == 3:
            head = torch.nn.Conv3d(
                architecture.num_out_channels, len(self.neighborhood), kernel_size=1
            )
        else:
            raise NotImplementedError(
                f"AffinitiesPredictor not implemented for {self.dims} dimensions"
            )

        return Model(architecture, head, eval_activation=torch.nn.Sigmoid())

    def create_target(self, gt: Array):
        """
        Create the target data.

        Args:
            gt: The ground truth data.
        Returns:
            NumpyArray: The created target data.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> predictor.create_target(gt)

        """
        # zeros
        assert np.prod(gt.physical_shape) == np.prod(gt.shape), (
            "Cannot create affinities from ground truth with nonspatial dimensions.\n"
            f"GT axis_names: {gt.axis_names}"
        )
        assert (
            gt.channel_dims <= 1
        ), "Cannot create affinities from ground truth with more than one channel dimension."
        label_data = gt[gt.roi]
        axis_names = gt.axis_names
        if gt.channel_dims == 1:
            label_data = label_data[0]
        else:
            axis_names = ["c^"] + axis_names
        affinities = seg_to_affgraph(
            label_data + int(self.background_as_object), self.neighborhood
        ).astype(np.float32)
        if self.lsds:
            descriptors = self.extractor(gt.voxel_size).get_descriptors(
                segmentation=label_data + int(self.background_as_object),
                voxel_size=gt.voxel_size,
            )
            return np_to_funlib_array(
                np.concatenate([affinities, descriptors], axis=0, dtype=np.float32),
                gt.roi.offset,
                gt.voxel_size,
            )
        return np_to_funlib_array(
            affinities,
            gt.roi.offset,
            gt.voxel_size,
        )

    def _grow_boundaries(self, mask, slab):
        """
        Grow the boundaries of the mask.

        Args:
            mask: The mask data.
            slab: The slab definition.
        Returns:
            np.ndarray: The mask with grown boundaries.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> predictor._grow_boundaries(mask, slab)
        """
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

    def create_weight(
        self, gt: Array, target: Array, mask: Array, moving_class_counts=None
    ):
        """
        Create the weight data.

        Args:
            gt: The ground truth data.
            target: The target data.
            mask: The mask data.
            moving_class_counts: The moving class counts.
        Returns:
            Tuple[NumpyArray, Tuple]: The created weight data and moving class counts.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> predictor.create_weight(gt, target, mask, moving_class_counts)
        """
        (moving_class_counts, moving_lsd_class_counts) = (
            moving_class_counts if moving_class_counts is not None else (None, None)
        )
        if self.grow_boundary_iterations > 0:
            mask_data = self._grow_boundaries(
                mask[target.roi],
                slab=tuple(1 if c == "c^" else -1 for c in target.axis_names),
            )
        else:
            mask_data = mask[target.roi]
        aff_weights, moving_class_counts = balance_weights(
            target[target.roi][: len(self.neighborhood)].astype(np.uint8),
            2,
            slab=tuple(1 if c == "c^" else -1 for c in target.axis_names),
            masks=[mask_data],
            moving_counts=moving_class_counts,
            clipmin=self.affs_weight_clipmin,
            clipmax=self.affs_weight_clipmax,
        )
        if self.lsds:
            lsd_weights, moving_lsd_class_counts = balance_weights(
                (gt[target.roi] > 0).astype(np.uint8),
                2,
                slab=(-1,) * len(gt.axis_names),
                masks=[mask_data],
                moving_counts=moving_lsd_class_counts,
                clipmin=self.lsd_weight_clipmin,
                clipmax=self.lsd_weight_clipmax,
            )
            lsd_weights = np.ones(
                (self.num_lsds,) + aff_weights.shape[1:], dtype=aff_weights.dtype
            ) * lsd_weights.reshape((1,) + aff_weights.shape[1:])
            return np_to_funlib_array(
                np.concatenate([aff_weights, lsd_weights], axis=0),
                target.roi.offset,
                target.voxel_size,
            ), (moving_class_counts, moving_lsd_class_counts)
        return np_to_funlib_array(
            aff_weights,
            target.roi.offset,
            target.voxel_size,
        ), (moving_class_counts, moving_lsd_class_counts)

    def gt_region_for_roi(self, target_spec):
        """
        Get the ground truth region for the target region of interest (ROI).

        Args:
            target_spec: The target region of interest (ROI) specification.
        Returns:
            The ground truth region specification.
        Raises:
            NotImplementedError: This method is not implemented.

        """
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
        """
        Get the output array type.

        Returns:
            EmbeddingArray: The output array type.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> predictor.output_array_type
        """
        return EmbeddingArray(self.dims)
