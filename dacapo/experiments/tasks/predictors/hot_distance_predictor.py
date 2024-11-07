from dacapo.experiments.arraytypes.probabilities import ProbabilityArray
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


class HotDistancePredictor(Predictor):
    """
    Predict signed distances and one hot embedding (as a proxy task) for a binary segmentation task.
    Distances deep within background are pushed to -inf, distances deep within
    the foreground object are pushed to inf. After distances have been
    calculated they are passed through a tanh so that distances saturate at +-1.
    Multiple classes can be predicted via multiple distance channels. The names
    of each class that is being segmented can be passed in as a list of strings
    in the channels argument.

    Attributes:
        channels: List of strings, each string is the name of a class that is being segmented.
        scale_factor: The scale factor for the distance transform.
        mask_distances: Whether to mask distances based on the distance to the boundary.
        norm: The normalization function to use for the distance transform.
        dt_scale_factor: The scale factor for the distance transform.
        max_distance: The maximum distance to consider for the distance transform.
        epsilon: The epsilon value to use for the distance transform.
        threshold: The threshold value to use for the distance transform.
    Methods:
        __init__(self, channels: List[str], scale_factor: float, mask_distances: bool): Initializes the HotDistancePredictor.
        create_model(self, architecture): Create the model for the predictor.
        create_target(self, gt): Create the target array for training.
        create_weight(self, gt, target, mask, moving_class_counts=None): Create the weight array for training.
        create_distance_mask(self, distances, mask, voxel_size, normalize=None, normalize_args=None): Create the distance mask for training.
        process(self, labels, voxel_size, normalize=None, normalize_args=None): Process the labels array and convert it to one-hot encoding.
        gt_region_for_roi(self, target_spec): Report how much spatial context this predictor needs to generate a target for the given ROI.
        padding(self, gt_voxel_size): Return the padding needed for the ground-truth
    Notes:
        This is a subclass of Predictor.
    """

    def __init__(self, channels: List[str], scale_factor: float, mask_distances: bool):
        """
        Initializes the HotDistancePredictor.

        Args:
            channels (List[str]): The list of class labels.
            scale_factor (float): The scale factor for the distance transform.
            mask_distances (bool): Whether to mask distances based on the distance to the boundary.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> predictor = HotDistancePredictor(channels, scale_factor, mask_distances)
        Note:
            The channels argument is a list of strings, each string is the name of a class that is being segmented.
        """
        self.channels = (
            channels * 2
        )  # one hot + distance (TODO: add hot/distance to channel names)
        self.norm = "tanh"
        self.dt_scale_factor = scale_factor
        self.mask_distances = mask_distances

        self.max_distance = 1 * scale_factor
        self.epsilon = 5e-2  # TODO: should be a config parameter
        self.threshold = 0.8  # TODO: should be a config parameter

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

    @property
    def classes(self):
        """
        Get the number of classes.

        Returns:
            int: The number of classes.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> classes = predictor.classes

        """
        return len(self.channels) // 2

    def create_model(self, architecture):
        """
        Create the model for the predictor.

        Args:
            architecture: The architecture for the model.
        Returns:
            Model: The created model.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> model = predictor.create_model(architecture)
        """
        if architecture.dims == 2:
            head = torch.nn.Conv2d(
                architecture.num_out_channels, self.embedding_dims, kernel_size=3
            )
        elif architecture.dims == 3:
            head = torch.nn.Conv3d(
                architecture.num_out_channels, self.embedding_dims, kernel_size=3
            )

        return Model(architecture, head)

    def create_target(self, gt):
        """
        Create the target array for training.

        Args:
            gt: The ground truth array.
        Returns:
            NumpyArray: The created target array.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> target = predictor.create_target(gt)
        """
        target = self.process(gt.data, gt.voxel_size, self.norm, self.dt_scale_factor)
        return np_to_funlib_array(
            target,
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
            Tuple[NumpyArray, np.ndarray]: The weight array and the moving class counts.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> predictor.create_weight(gt, target, mask, moving_class_counts)
        """
        # balance weights independently for each channel
        one_hot_weights, one_hot_moving_class_counts = balance_weights(
            gt[target.roi],
            2,
            slab=tuple(1 if c == "c^" else -1 for c in gt.axis_names),
            masks=[mask[target.roi]],
            moving_counts=(
                None
                if moving_class_counts is None
                else moving_class_counts[: self.classes]
            ),
        )

        if self.mask_distances:
            distance_mask = self.create_distance_mask(
                target[target.roi][-self.classes :],
                mask[target.roi],
                target.voxel_size,
                self.norm,
                self.dt_scale_factor,
            )
        else:
            distance_mask = np.ones_like(gt.data)

        distance_weights, distance_moving_class_counts = balance_weights(
            gt[target.roi],
            2,
            slab=tuple(1 if c == "c^" else -1 for c in gt.axis_names),
            masks=[mask[target.roi], distance_mask],
            moving_counts=(
                None
                if moving_class_counts is None
                else moving_class_counts[-self.classes :]
            ),
        )

        weights = np.concatenate((one_hot_weights, distance_weights))
        moving_class_counts = np.concatenate(
            (one_hot_moving_class_counts, distance_moving_class_counts)
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
            ProbabilityArray: The output array type.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> output_array_type = predictor.output_array_type
        Notes:
            Technically this is a probability array + distance array, but it is only ever referenced for interpolatability (which is true for both).
        """
        # technically this is a probability array + distance array, but it is only ever referenced for interpolatability (which is true for both) (TODO)
        return ProbabilityArray(self.embedding_dims)

    def create_distance_mask(
        self,
        distances: np.ndarray,
        mask: np.ndarray,
        voxel_size: Coordinate,
        normalize=None,
        normalize_args=None,
    ):
        """
        Create the distance mask for training.

        Args:
            distances: The distances array.
            mask: The mask array.
            voxel_size: The voxel size.
            normalize: The normalization function to use.
            normalize_args: The normalization arguments.
        Returns:
            np.ndarray: The distance mask.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> distance_mask = self.create_distance_mask(distances, mask, voxel_size, normalize, normalize_args)
        """
        mask_output = mask.copy()
        for i, (channel_distance, channel_mask) in enumerate(zip(distances, mask)):
            tmp = np.zeros(
                np.array(channel_mask.shape) + np.array((2,) * channel_mask.ndim),
                dtype=channel_mask.dtype,
            )
            slices = tmp.ndim * (slice(1, -1),)
            tmp[slices] = channel_mask
            boundary_distance = distance_transform_edt(
                tmp,
                sampling=voxel_size,
            )
            if self.epsilon is None:
                add = 0
            else:
                add = self.epsilon
            boundary_distance = self.__normalize(
                boundary_distance[slices], normalize, normalize_args
            )

            channel_mask_output = mask_output[i]
            logging.debug(
                "Total number of masked in voxels before distance masking {0:}".format(
                    np.sum(channel_mask_output)
                )
            )
            channel_mask_output[
                np.logical_and(
                    np.clip(abs(channel_distance) + add, 0, self.threshold)
                    >= boundary_distance,
                    channel_distance >= 0,
                )
            ] = 0
            logging.debug(
                "Total number of masked in voxels after postive distance masking {0:}".format(
                    np.sum(channel_mask_output)
                )
            )
            channel_mask_output[
                np.logical_and(
                    np.clip(abs(channel_distance) + add, 0, self.threshold)
                    >= boundary_distance,
                    channel_distance <= 0,
                )
            ] = 0
            logging.debug(
                "Total number of masked in voxels after negative distance masking {0:}".format(
                    np.sum(channel_mask_output)
                )
            )
        return mask_output

    def process(
        self,
        labels: np.ndarray,
        voxel_size: Coordinate,
        normalize=None,
        normalize_args=None,
    ):
        """
        Process the labels array and convert it to one-hot encoding.

        Args:
            labels: The labels array.
            voxel_size: The voxel size.
            normalize: The normalization function to use.
            normalize_args: The normalization arguments.
        Returns:
            np.ndarray: The one-hot encoded array.
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

        return np.concatenate((labels, all_distances))

    def __find_boundaries(self, labels):
        """
        Find the boundaries in the labels array.

        Args:
            labels: The labels array.
        Returns:
            The boundaries array.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> boundaries = self.__find_boundaries(labels)
        Notes:
            Assumes labels has a singleton channel dim and channel dim is first.
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
        Normalize the distances.

        Args:
            distances: The distances to normalize.
            norm: The normalization function to use.
            normalize_args: The normalization arguments.
        Returns:
            The normalized distances.
        Raises:
            ValueError: Only tanh is supported for normalization.
        Examples:
            >>> normalized_distances = self.__normalize(distances, norm, normalize_args)
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
            Coordinate: The padding needed for the ground-truth array.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> predictor.padding(gt_voxel_size)
        """
        return Coordinate((self.max_distance,) * gt_voxel_size.dims)
