from .predictor import Predictor
from dacapo.experiments import Model
from dacapo.experiments.arraytypes import DistanceArray
from dacapo.tmp import np_to_funlib_array
from dacapo.utils.balance_weights import balance_weights

from funlib.geometry import Coordinate

from scipy.ndimage.morphology import distance_transform_edt
import numpy as np
import torch

import logging
from typing import List

logger = logging.getLogger(__name__)


class InnerDistancePredictor(Predictor):
    """
    Predict signed distances for a binary segmentation task.

    Distances deep within background are pushed to -inf, distances deep within
    the foreground object are pushed to inf. After distances have been
    calculated they are passed through a tanh so that distances saturate at +-1.
    Multiple classes can be predicted via multiple distance channels. The names
    of each class that is being segmented can be passed in as a list of strings
    in the channels argument.

    Attributes:
        channels (List[str]): The list of channel names.
        scale_factor (float): The amount by which to scale distances before applying a tanh normalization.
    Methods:
        __init__(self, channels: List[str], scale_factor: float): Initializes the InnerDistancePredictor.
        create_model(self, architecture): Create the model for the predictor.
        create_target(self, gt): Create the target array for training.
        create_weight(self, gt, target, mask, moving_class_counts=None): Create the weight array for training.
        output_array_type: Get the output array type.
        process(self, labels: np.ndarray, voxel_size: Coordinate, normalize=None, normalize_args=None): Process the labels array and convert it to signed distances.
        __find_boundaries(self, labels): Find the boundaries in a labels array.
        __normalize(self, distances, norm, normalize_args): Normalize the distances.
        gt_region_for_roi(self, target_spec): Get the ground-truth region for the given ROI.
        padding(self, gt_voxel_size: Coordinate) -> Coordinate: Get the padding needed for the ground-truth array.
    Notes:
        This is a subclass of Predictor.
    """

    def __init__(self, channels: List[str], scale_factor: float):
        """
        Initialize the InnerDistancePredictor.

        Args:
            channels (List[str]): The list of channel names.
            scale_factor (float): The amount by which to scale distances before applying a tanh normalization.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> predictor = InnerDistancePredictor(channels, scale_factor)
        """
        self.channels = channels
        self.norm = "tanh"
        self.dt_scale_factor = scale_factor

        self.max_distance = 1 * scale_factor
        self.epsilon = 5e-2
        self.threshold = 0.8

    @property
    def embedding_dims(self):
        """
        Get the number of embedding dimensions.

        Returns:
            int: The number of embedding dimensions.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> embedding_dims = predictor.embedding_dims
        """
        return len(self.channels)

    def create_model(self, architecture):
        """
        Create the model for the predictor.

        Args:
            architecture: The architecture for the model.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> model = predictor.create_model(architecture)
        """
        if architecture.dims == 2:
            head = torch.nn.Conv2d(
                architecture.num_out_channels, self.embedding_dims, kernel_size=1
            )
        elif architecture.dims == 3:
            head = torch.nn.Conv3d(
                architecture.num_out_channels, self.embedding_dims, kernel_size=1
            )

        return Model(architecture, head)

    def create_target(self, gt):
        """
        Create the target array for training.

        Args:
            gt: The ground-truth array.
        Returns:
            The DistanceArray.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> predictor.create_target(gt)

        """
        distances = self.process(
            gt.data, gt.voxel_size, self.norm, self.dt_scale_factor
        )
        return np_to_funlib_array(
            distances,
            gt.roi,
            gt.voxel_size,
            gt.axis_names,
        )

    def create_weight(self, gt, target, mask, moving_class_counts=None):
        """
        Create the weight array for training, given a ground-truth and
        associated target array.

        Args:
            gt: The ground-truth array.
            target: The target array.
            mask: The mask array.
            moving_class_counts: The moving class counts.
        Returns:
            The weight array and the moving class counts.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> predictor.create_weight(gt, target, mask, moving_class_counts)

        """
        # balance weights independently for each channel

        weights, moving_class_counts = balance_weights(
            gt[target.roi],
            2,
            slab=tuple(1 if c == "c^" else -1 for c in gt.axis_names),
            masks=[mask[target.roi]],
            moving_counts=moving_class_counts,
        )
        return (
            np_to_funlib_array(
                weights,
                gt.roi,
                gt.voxel_size,
                gt.axis_names,
            ),
            moving_class_counts,
        )

    @property
    def output_array_type(self):
        """
        Get the output array type.

        Returns:
            The DistanceArray.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> predictor.output_array_type

        """
        return DistanceArray(self.embedding_dims)

    def process(
        self,
        labels: np.ndarray,
        voxel_size: Coordinate,
        normalize=None,
        normalize_args=None,
    ):
        """
        Process the labels array and convert it to signed distances.

        Args:
            labels: The labels array.
            voxel_size: The voxel size.
            normalize: The normalization method.
            normalize_args: The normalization arguments.
        Returns:
            The signed distances.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> predictor.process(labels, voxel_size, normalize, normalize_args)

        """
        all_distances = np.zeros(labels.shape, dtype=np.float32) - 1
        for ii, channel in enumerate(labels):
            boundaries = self.__find_boundaries(channel)

            # mark boundaries with 0 (not 1)
            boundaries = 1.0 - boundaries

            if np.sum(boundaries == 0) == 0:
                max_distance = min(
                    dim * vs / 2 for dim, vs in zip(channel.shape, voxel_size)
                )
                if np.sum(channel) == 0:
                    distances = -np.ones(channel.shape, dtype=np.float32) * max_distance
                else:
                    distances = np.ones(channel.shape, dtype=np.float32) * max_distance
            else:
                # get distances (voxel_size/2 because image is doubled)
                distances = distance_transform_edt(
                    boundaries, sampling=tuple(float(v) / 2 for v in voxel_size)
                )
                distances = distances.astype(np.float32)

                # restore original shape
                downsample = (slice(None, None, 2),) * len(voxel_size)
                distances = distances[downsample]

                # todo: inverted distance
                distances[channel == 0] = -distances[channel == 0]

            if normalize is not None:
                distances = self.__normalize(distances, normalize, normalize_args)

            all_distances[ii] = distances

        return all_distances * labels

    def __find_boundaries(self, labels):
        """
        Find boundaries in a labels array.

        Args:
            labels: The labels array.
        Returns:
            The boundaries array.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> predictor.__find_boundaries(labels)

        """
        # labels: 1 1 1 1 0 0 2 2 2 2 3 3       n
        # shift :   1 1 1 1 0 0 2 2 2 2 3       n - 1
        # diff  :   0 0 0 1 0 1 0 0 0 1 0       n - 1
        # bound.: 00000001000100000001000      2n - 1

        logger.debug(f"computing boundaries for {labels.shape}")

        dims = len(labels.shape)
        in_shape = labels.shape
        out_shape = tuple(2 * s - 1 for s in in_shape)

        boundaries = np.zeros(out_shape, dtype=bool)

        logger.debug(f"boundaries shape is {boundaries.shape}")

        for d in range(dims):
            logger.debug(f"processing dimension {d}")

            shift_p = [slice(None)] * dims
            shift_p[d] = slice(1, in_shape[d])

            shift_n = [slice(None)] * dims
            shift_n[d] = slice(0, in_shape[d] - 1)

            diff = (labels[tuple(shift_p)] - labels[tuple(shift_n)]) != 0

            logger.debug(f"diff shape is {diff.shape}")

            target = [slice(None, None, 2)] * dims
            target[d] = slice(1, out_shape[d], 2)

            logger.debug(f"target slices are {target}")

            boundaries[tuple(target)] = diff

        return boundaries

    def __normalize(self, distances, norm, normalize_args):
        """
        Normalize distances.

        Args:
            distances: The distances to normalize.
            norm: The normalization method.
            normalize_args: The normalization arguments.
        Returns:
            The normalized distances.
        Raises:
            ValueError: If the normalization method is not supported.
        Examples:
            >>> predictor.__normalize(distances, norm, normalize_args)
        """
        if norm == "tanh":
            scale = normalize_args
            return np.tanh(distances / scale)
        else:
            raise ValueError("Only tanh is supported for normalization")

    def gt_region_for_roi(self, target_spec):
        """
        Report how much spatial context this predictor needs to generate a
        target for the given ROI. By default, uses the same ROI.

        Args:
            target_spec: The ROI for which the target is requested.
        Returns:
            The ROI for which the ground-truth is requested.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> predictor.gt_region_for_roi(target_spec)

        """
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
        """
        Return the padding needed for the ground-truth array.

        Args:
            gt_voxel_size: The voxel size of the ground-truth array.
        Returns:
            The padding needed for the ground-truth array.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> predictor.padding(gt_voxel_size)
        """
        return Coordinate((self.max_distance,) * gt_voxel_size.dims)
