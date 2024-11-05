from .predictor import Predictor
from dacapo.experiments import Model
from dacapo.experiments.arraytypes import DistanceArray
from dacapo.utils.balance_weights import balance_weights
from dacapo.tmp import np_to_funlib_array

from funlib.geometry import Coordinate
from funlib.persistence import Array

from scipy.ndimage.morphology import distance_transform_edt
import numpy as np
import torch

import logging
from typing import List

logger = logging.getLogger(__name__)


class DistancePredictor(Predictor):
    """
    Predict signed distances for a binary segmentation task.
    Distances deep within background are pushed to -inf, distances deep within
    the foreground object are pushed to inf. After distances have been
    calculated they are passed through a tanh so that distances saturate at +-1.
    Multiple classes can be predicted via multiple distance channels. The names
    of each class that is being segmented can be passed in as a list of strings
    in the channels argument.

    Attributes:
        channels (List[str]): The list of class labels.
        scale_factor (float): The scale factor for the distance transform.
        mask_distances (bool): Whether to mask distances.
        clipmin (float): The minimum value to clip the weights to.
        clipmax (float): The maximum value to clip the weights to.
    Methods:
        __init__(self, channels: List[str], scale_factor: float, mask_distances: bool, clipmin: float = 0.05, clipmax: float = 0.95): Initializes the DistancePredictor.
        create_model(self, architecture): Create the model for the predictor.
        create_target(self, gt): Create the target array for training.
        create_weight(self, gt, target, mask, moving_class_counts=None): Create the weight array for training.
        output_array_type: Get the output array type.
        create_distance_mask(self, distances, mask, voxel_size, normalize=None, normalize_args=None): Create the distance mask.
        process(self, labels, voxel_size, normalize=None, normalize_args=None): Process the labels array.
        gt_region_for_roi(self, target_spec): Get the ground-truth region for the ROI.
        padding(self, gt_voxel_size: Coordinate) -> Coordinate: Get the padding needed for the ground-truth array.
    Notes:
        The DistancePredictor is used to predict signed distances for a binary segmentation task.
        The distances are calculated using the distance_transform_edt function from scipy.ndimage.morphology.
        The distances are then passed through a tanh function to saturate the distances at +-1.
        The distances are calculated for each class that is being segmented and are stored in separate channels.
        The names of each class that is being segmented can be passed in as a list of strings in the channels argument.
        The scale_factor argument is used to scale the distances.
        The mask_distances argument is used to determine whether to mask distances.
        The clipmin argument is used to determine the minimum value to clip the weights to.
        The clipmax argument is used to determine the maximum value to clip the weights to.
    """

    def __init__(
        self,
        channels: List[str],
        scale_factor: float,
        mask_distances: bool,
        clipmin: float = 0.05,
        clipmax: float = 0.95,
    ):
        """
        Initialize the DistancePredictor object.

        Args:
            channels (List[str]): List of channel names.
            scale_factor (float): Scale factor for distance calculation.
            mask_distances (bool): Flag indicating whether to mask distances.
            clipmin (float, optional): Minimum clipping value. Defaults to 0.05.
            clipmax (float, optional): Maximum clipping value. Defaults to 0.95.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> predictor = DistancePredictor(channels, scale_factor, mask_distances, clipmin, clipmax)
        """
        self.channels = channels
        self.norm = "tanh"
        self.dt_scale_factor = scale_factor
        self.mask_distances = mask_distances

        self.max_distance = 1 * scale_factor
        self.epsilon = 5e-2
        self.threshold = 0.8
        self.clipmin = clipmin
        self.clipmax = clipmax

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
        Returns:
            Model: The created model.
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

    def create_target(self, gt: Array):
        """
        Turn the ground truth labels into a distance transform.
        """
        distances = self.process(gt[:], gt.voxel_size, self.norm, self.dt_scale_factor)
        return np_to_funlib_array(
            distances,
            gt.roi.offset,
            gt.voxel_size,
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
        if self.mask_distances:
            distance_mask = self.create_distance_mask(
                target[target.roi],
                mask[target.roi],
                target.voxel_size,
                self.norm,
                self.dt_scale_factor,
            )
        else:
            distance_mask = np.ones_like(target.data)

        weights, moving_class_counts = balance_weights(
            gt[target.roi],
            2,
            slab=tuple(1 if c == "c^" else -1 for c in gt.axis_names),
            masks=[mask[target.roi], distance_mask],
            moving_counts=moving_class_counts,
            clipmin=self.clipmin,
            clipmax=self.clipmax,
        )
        return (
            np_to_funlib_array(
                weights,
                gt.roi.offset,
                gt.voxel_size,
            ),
            moving_class_counts,
        )

    @property
    def output_array_type(self):
        """
        Get the output array type.

        Returns:
            DistanceArray: The output array type.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> predictor.output_array_type
        """
        return DistanceArray(self.embedding_dims)

    def create_distance_mask(
        self,
        distances: np.ndarray,
        mask: np.ndarray,
        voxel_size: Coordinate,
        normalize=None,
        normalize_args=None,
    ):
        """
        Create a distance mask.

        Args:
            distances (np.ndarray): The distances array.
            mask (np.ndarray): The mask array.
            voxel_size (Coordinate): The voxel size.
            normalize (str): The normalization method.
            normalize_args: The normalization arguments.
        Returns:
            np.ndarray: The distance mask.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> predictor.create_distance_mask(distances, mask, voxel_size, normalize, normalize_args)

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
            labels (np.ndarray): The labels array.
            voxel_size (Coordinate): The voxel size.
            normalize (str): The normalization method.
            normalize_args: The normalization arguments.
        Returns:
            np.ndarray: The distances array.
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

        return all_distances

    def __find_boundaries(self, labels: np.ndarray):
        """
        Find the boundaries in the labels.

        Args:
            labels: The labels.
        Returns:
            The boundaries.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> predictor.__find_boundaries(labels)

        """
        # labels: 1 1 1 1 0 0 2 2 2 2 3 3       n
        # shift :   1 1 1 1 0 0 2 2 2 2 3       n - 1
        # diff  :   0 0 0 1 0 1 0 0 0 1 0       n - 1
        # bound.: 00000001000100000001000      2n - 1

        if labels.dtype == bool:
            raise ValueError("Labels should not be bools")
            labels = labels.astype(np.uint8)

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
            gt_voxel_size (Coordinate): The voxel size of the ground-truth array.
        Returns:
            Coordinate: The padding needed for the ground-truth array.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> predictor.padding(gt_voxel_size)

        """
        return Coordinate((self.max_distance,) * gt_voxel_size.dims)
