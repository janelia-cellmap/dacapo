from .predictor import Predictor
from dacapo.experiments import Model
from dacapo.experiments.arraytypes import DistanceArray
from dacapo.experiments.datasplits.datasets.arrays import NumpyArray

from funlib.geometry import Coordinate

from scipy.ndimage.morphology import distance_transform_edt
import numpy as np
import torch

from typing import List
import logging

logger = logging.getLogger(__file__)


class DistancePredictor(Predictor):
    def __init__(self, channels: List[str], scale_factor: float):
        self.channels = channels
        self.norm = "tanh"
        self.dt_scale_factor = scale_factor

    @property
    def embedding_dims(self):
        return len(self.channels)

    def create_model(self, architecture):

        head = torch.nn.Conv3d(
            architecture.num_out_channels, self.embedding_dims, kernel_size=3
        )

        return Model(architecture, head)

    def create_target(self, gt):
        distances = self.process(
            gt.data, gt.voxel_size, self.norm, self.dt_scale_factor
        )
        return NumpyArray.from_np_array(
            distances,
            gt.roi,
            gt.voxel_size,
            gt.axes,
        )

    def create_weight(self, gt, target):
        return NumpyArray.from_np_array(
            np.ones(target.data.shape),
            target.roi,
            target.voxel_size,
            target.axes,
        )

    @property
    def output_array_type(self):
        return DistanceArray(self.embedding_dims)

    def process(
        self,
        labels: np.ndarray,
        voxel_size: Coordinate,
        normalize=None,
        normalize_args=None,
    ):

        all_distances = np.zeros(labels.shape, dtype=np.float32) - 1
        for ii, channel in enumerate(labels):
            boundaries = self.__find_boundaries(channel)

            # mark boundaries with 0 (not 1)
            boundaries = 1.0 - boundaries

            if np.sum(boundaries == 0) == 0:
                max_distance = min(
                    dim * vs for dim, vs in zip(channel.shape, voxel_size)
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

    def __find_boundaries(self, labels):

        # labels: 1 1 1 1 0 0 2 2 2 2 3 3       n
        # shift :   1 1 1 1 0 0 2 2 2 2 3       n - 1
        # diff  :   0 0 0 1 0 1 0 0 0 1 0       n - 1
        # bound.: 00000001000100000001000      2n - 1

        logger.debug("computing boundaries for %s", labels.shape)

        dims = len(labels.shape)
        in_shape = labels.shape
        out_shape = tuple(2 * s - 1 for s in in_shape)
        out_slices = tuple(slice(0, s) for s in out_shape)

        boundaries = np.zeros(out_shape, dtype=np.bool)

        logger.debug("boundaries shape is %s", boundaries.shape)

        for d in range(dims):

            logger.debug("processing dimension %d", d)

            shift_p = [slice(None)] * dims
            shift_p[d] = slice(1, in_shape[d])

            shift_n = [slice(None)] * dims
            shift_n[d] = slice(0, in_shape[d] - 1)

            diff = (labels[tuple(shift_p)] - labels[tuple(shift_n)]) != 0

            logger.debug("diff shape is %s", diff.shape)

            target = [slice(None, None, 2)] * dims
            target[d] = slice(1, out_shape[d], 2)

            logger.debug("target slices are %s", target)

            boundaries[tuple(target)] = diff

        return boundaries

    def __normalize(self, distances, norm, normalize_args):

        if norm == "tanh":
            scale = normalize_args
            return np.tanh(distances / scale)
        else:
            raise ValueError("Only tanh is supported for normalization")
