from __future__ import division

import logging
import math

import scipy.ndimage
from scipy.spatial.transform import Rotation as R

import numpy as np

import augment

from gunpowder import BatchFilter, Roi, ArrayKey, Coordinate, GraphKey

logger = logging.getLogger(__name__)


def _create_identity_transformation(shape, voxel_size=None, offset=None, subsample=1):
    
    dims = len(shape)

    if voxel_size is None:
        voxel_size = Coordinate((1,) * dims)

    if offset is None:
        offset = Coordinate((0,) * dims)
    subsample_shape = tuple(max(1, int(s / subsample)) for s in shape)
    step_width = tuple(
        float(shape[d] - 1) / (subsample_shape[d] - 1) if subsample_shape[d] > 1 else 1
        for d in range(dims)
    )
    step_width = tuple(s * vs for s, vs in zip(step_width, voxel_size))

    axis_ranges = (
        np.arange(subsample_shape[d], dtype=np.float32) * step_width[d] + offset[d]
        for d in range(dims)
    )
    return np.array(np.meshgrid(*axis_ranges, indexing="ij"), dtype=np.float32)


def _upscale_transformation(
    transformation, output_shape, interpolate_order=1, dtype=np.float32
):
    

    input_shape = transformation.shape[1:]

    dims = len(output_shape)
    scale = tuple(float(s) / c for s, c in zip(output_shape, input_shape))

    scaled = np.empty((dims,) + output_shape, dtype=dtype)
    for d in range(dims):
        scipy.ndimage.zoom(
            transformation[d],
            zoom=scale,
            output=scaled[d],
            order=interpolate_order,
            mode="nearest",
        )

    return scaled


def _rotate(point, angle):
    
    res = np.array(point)
    res[0] = math.sin(angle) * point[1] + math.cos(angle) * point[0]
    res[1] = -math.sin(angle) * point[0] + math.cos(angle) * point[1]

    return res


def _create_rotation_transformation(shape, angle, subsample=1, voxel_size=None):
    
    dims = len(shape)
    subsample_shape = tuple(max(1, int(s / subsample)) for s in shape)
    control_points = (2,) * dims

    if voxel_size is None:
        voxel_size = Coordinate((1,) * dims)

    # map control points to world coordinates
    control_point_scaling_factor = tuple(
        float(s - 1) * vs for s, vs in zip(shape, voxel_size)
    )

    # rotate control points
    center = np.array([0.5 * (d - 1) * vs for d, vs in zip(shape, voxel_size)])

    # print("Creating rotation transformation with:")
    # print("\tangle : " + str(angle))
    # print("\tcenter: " + str(center))

    control_point_offsets = np.zeros((dims,) + control_points, dtype=np.float32)
    for control_point in np.ndindex(control_points):
        point = np.array(control_point) * control_point_scaling_factor
        center_offset = np.array(
            [p - c for c, p in zip(center, point)], dtype=np.float32
        )
        rotated_offset = np.array(center_offset)
        rotated_offset[-2:] = _rotate(center_offset[-2:], angle)
        displacement = rotated_offset - center_offset
        control_point_offsets[(slice(None),) + control_point] += displacement

    return augment.upscale_transformation(control_point_offsets, subsample_shape)


def _create_uniform_3d_transformation(shape, rotation, subsample=1, voxel_size=None):
    
    dims = len(shape)
    subsample_shape = tuple(max(1, int(s / subsample)) for s in shape)
    control_points = (2,) * dims

    if voxel_size is None:
        voxel_size = Coordinate((1,) * dims)

    # map control points to world coordinates
    control_point_scaling_factor = tuple(
        float(s - 1) * vs for s, vs in zip(shape, voxel_size)
    )

    # rotate control points
    center = np.array([0.5 * (d - 1) * vs for d, vs in zip(shape, voxel_size)])

    # print("Creating rotation transformation with:")
    # print("\tangle : " + str(angle))
    # print("\tcenter: " + str(center))

    control_point_offsets = np.zeros((dims,) + control_points, dtype=np.float32)
    for control_point in np.ndindex(control_points):
        point = np.array(control_point) * control_point_scaling_factor
        center_offset = np.array(
            [p - c for c, p in zip(center, point)], dtype=np.float32
        )
        rotated_offset = np.array(center_offset)
        rotated_offset = rotation.apply(rotated_offset)
        displacement = rotated_offset - center_offset
        control_point_offsets[(slice(None),) + control_point] += displacement

    return augment.upscale_transformation(control_point_offsets, subsample_shape)


def _min_max_mean_std(ndarray, prefix=""):
    
    return ""


class ElasticAugment(BatchFilter):
    

    def __init__(
        self,
        control_point_spacing,
        control_point_displacement_sigma,
        rotation_interval,
        subsample=1,
        augmentation_probability=1.0,
        seed=None,
        uniform_3d_rotation=False,
    ):
        
        super(BatchFilter, self).__init__()
        self.control_point_spacing = control_point_spacing
        self.control_point_displacement_sigma = control_point_displacement_sigma
        self.rotation_start = rotation_interval[0]
        self.rotation_max_amount = rotation_interval[1] - rotation_interval[0]
        self.subsample = subsample
        self.augmentation_probability = augmentation_probability
        self.uniform_3d_rotation = uniform_3d_rotation
        self.do_augment = False

        logger.debug(
            f"initialized with parameters "
            f"control_point_spacing={self.control_point_spacing} "
            f"control_point_displacement_sigma={self.control_point_displacement_sigma} "
            f"rotation_start={self.rotation_start} "
            f"rotation_max_amount={self.rotation_max_amount} "
            f"subsample={self.subsample} "
            f"seed={seed}"
        )

        assert isinstance(self.subsample, int), "subsample has to be integer"
        assert self.subsample >= 1, "subsample has to be strictly positive"

        self.transformations = {}
        self.target_rois = {}

    def setup(self):
        
        self.voxel_size = Coordinate(
            min(axis)
            for axis in zip(
                *[
                    array_spec.voxel_size
                    for array_spec in self.spec.array_specs.values()
                ]
            )
        )
        self.spatial_dims = self.voxel_size.dims

    def prepare(self, request):
        
        logger.debug(
            logger.debug(
                f"{type(self).__name__} preparing request {request} with transformation voxel size {self.voxel_size}"
            )
        )

        total_roi = request.get_total_roi()
        master_roi = self._spatial_roi(total_roi)
        logger.debug(f"master roi is {master_roi} with voxel size {self.voxel_size}")

        uniform_random_sample = np.random.rand()
        logger.debug(
            f"Prepare: Uniform random sample is {uniform_random_sample}, probability to augment is {self.augmentation_probability}",
        )
        self.do_augment = uniform_random_sample < self.augmentation_probability
        if not self.do_augment:
            logger.debug(
                logger.debug(
                    f"Prepare: Randomly not augmenting at all. (probability to augment: {self.augmentation_probability})"
                )
            )
            return

        master_roi_snapped = master_roi.snap_to_grid(self.voxel_size, mode="grow")
        master_roi_voxels = master_roi_snapped // self.voxel_size
        master_transform = self._create_transformation(
            master_roi_voxels.get_shape(), offset=master_roi_snapped.get_begin()
        )

        self.transformations.clear()
        self.target_rois.clear()

        logger.debug(
            f"Master transformation statistics: {_min_max_mean_std(master_transform)}"
        )

        for key, spec in request.items():
            assert isinstance(key, ArrayKey) or isinstance(
                key, GraphKey
            ), f"Only ArrayKey/GraphKey supported but got {type(key)} in request"

            logger.debug(f"key {key}: preparing with spec {spec}")

            if isinstance(key, ArrayKey):
                voxel_size = self.spec[key].voxel_size
            else:
                voxel_size = Coordinate((1,) * spec.roi.dims)
            # Todo we could probably remove snap_to_grid, we already check spec.roi % voxel_size == 0

            target_roi = spec.roi.snap_to_grid(voxel_size)

            self.target_rois[key] = target_roi
            target_roi_voxels = target_roi // voxel_size

            # get scale and offset to transform/interpolate master displacement to current spec
            vs_ratio = np.array(
                [vs1 / vs2 for vs1, vs2 in zip(voxel_size, self.voxel_size)]
            )
            offset_world = target_roi.get_begin() - master_roi_snapped.get_begin()
            scale = vs_ratio
            offset = offset_world / self.voxel_size

            logger.debug(f"key {key}: scale {scale} and offset {offset}")

            # need to pass inverse transform, hence -offset
            transform = self._affine(master_transform, scale, offset, target_roi_voxels)
            logger.debug(
                logger.debug(
                    f"key {key}: transformed transform statistics {_min_max_mean_std(transform)}"
                )
            )
            source_roi = self._get_source_roi(transform).snap_to_grid(voxel_size)
            logger.debug(
                logger.debug(
                    f"key {key}: source roi (target roi) is {source_roi} ({target_roi})"
                )
            )
            self._shift_transformation(-target_roi.get_begin(), transform)
            logger.debug(
                logger.debug(
                    f"key {key}: shifted transformed transform statistics: {_min_max_mean_std(transform)}"
                )
            )
            for d, (vs, b1, b2) in enumerate(
                zip(voxel_size, target_roi.get_begin(), source_roi.get_begin())
            ):
                pixel_offset = (b1 - b2) / vs
                transform[d] = transform[d] / vs + pixel_offset
            logger.debug(
                logger.debug(
                    f"key {key}: pixel-space transform statistics: {_min_max_mean_std(transform)}"
                )
            )

            self.transformations[key] = transform

            # update upstream request
            spec.roi = Roi(
                spec.roi.get_begin()[: -self.spatial_dims]
                + source_roi.get_begin()[-self.spatial_dims :],
                spec.roi.get_shape()[: -self.spatial_dims]
                + source_roi.get_shape()[-self.spatial_dims :],
            )

    def process(self, batch, request):
        
        if not self.do_augment:
            logger.debug(
                f"Process: Randomly not augmenting at all. (probability to augment: {self.augmentation_probability})"
            )
            return

        for key, _ in request.items():
            if isinstance(key, GraphKey):
                # restore original ROIs
                logger.warning("GRAPHS NOT PROPERLY SUPPORTED!")
                batch[key].spec.roi = request[key].roi
                continue

            assert key in batch.arrays, f"only arrays supported but got {key}"
            array = batch.arrays[key]

            # for arrays, the target ROI and the requested ROI should be the
            # same in spatial coordinates
            assert (
                self.target_rois[key].get_begin()
                == request[key].roi.get_begin()[-self.spatial_dims :]
            ), "inconsistent offsets {} -- {} for key {}".format(
                self.target_rois[key].get_begin(),
                request[key].roi.get_begin()[-self.spatial_dims :],
                key,
            )
            assert (
                self.target_rois[key].get_shape()
                == request[key].roi.get_shape()[-self.spatial_dims :]
            )

            # reshape array data into (channels,) + spatial dims
            shape = array.data.shape
            data = array.data.reshape((-1,) + shape[-self.spatial_dims :])
            logger.debug(
                logger.debug(
                    f"key {key}: applying transform with statistics {tuple(map(np.mean, self.transformations[key]))} {tuple(map(np.std, self.transformations[key]))}"
                )
            )

            # apply transformation on each channel
            data = np.array(
                [
                    augment.apply_transformation(
                        data[c],
                        self.transformations[key],
                        interpolate=self.spec[key].interpolatable,
                    )
                    for c in range(data.shape[0])
                ]
            )

            data_roi = request[key].roi / self.spec[key].voxel_size
            array.data = data.reshape(
                array.data.shape[: -self.spatial_dims] + data_roi.get_shape()
            )

            # restore original ROIs
            array.spec.roi = request[key].roi

    def _create_transformation(self, target_shape, offset):
        
        logger.debug(
            f"creating displacement for shape {target_shape}, subsample {self.subsample}",
        )
        transformation = _create_identity_transformation(
            target_shape,
            subsample=self.subsample,
            voxel_size=self.voxel_size,
            offset=offset,
        )
        if np.any(np.asarray(self.control_point_displacement_sigma) > 0):
            logger.debug(
                f"Jittering with sigma={self.control_point_displacement_sigma} and spacing={self.control_point_spacing}",
            )
            elastic = augment.create_elastic_transformation(
                target_shape,
                self.control_point_spacing,
                self.control_point_displacement_sigma,
                subsample=self.subsample,
            )
            logger.debug(
                f"elastic displacements statistics: {_min_max_mean_std(elastic)}"
            )
            transformation += elastic
        if not self.uniform_3d_rotation:
            rotation = (
                np.random.random() * self.rotation_max_amount + self.rotation_start
            )
            if rotation != 0:
                logger.debug(f"rotating with rotation={rotation}")
                transformation += _create_rotation_transformation(
                    target_shape,
                    rotation,
                    voxel_size=self.voxel_size,
                    subsample=self.subsample,
                )
        else:
            rotation = R.random()
            transformation += _create_uniform_3d_transformation(
                target_shape,
                rotation,
                voxel_size=self.voxel_size,
                subsample=self.subsample,
            )

        if self.subsample > 1:
            logger.debug(
                f"transform statistics before upscale: {_min_max_mean_std(transformation)}",
            )
            transformation = _upscale_transformation(transformation, target_shape)
            logger.debug(
                f"transform statistics after upscale: {_min_max_mean_std(transformation)}",
            )

        return transformation

    def _spatial_roi(self, roi):
        
        return Roi(
            roi.get_begin()[-self.spatial_dims :], roi.get_shape()[-self.spatial_dims :]
        )

    def _affine(self, array, scale, offset, target_roi, dtype=np.float32, order=1):
        
        ndim = array.shape[0]
        output = np.empty((ndim,) + target_roi.get_shape(), dtype=dtype)
        # Create a diagonal matrix if scale is a 1-D array
        if np.isscalar(scale) or np.ndim(scale) == 1:
            transform_matrix = np.diag(scale)
        else:
            transform_matrix = scale
        for d in range(ndim):
            scipy.ndimage.affine_transform(
                input=array[d],
                matrix=transform_matrix,
                offset=offset,
                output=output[d],
                output_shape=output[d].shape,
                order=order,
                mode="nearest",
            )
        return output

    def _shift_transformation(self, shift, transformation):
        
        for d in range(transformation.shape[0]):
            transformation[d] += shift[d]

    def _get_source_roi(self, transformation):
        
        dims = transformation.shape[0]

        # get bounding box of needed data for transformation
        bb_min = Coordinate(
            int(math.floor(transformation[d].min())) for d in range(dims)
        )
        bb_max = Coordinate(
            int(math.ceil(transformation[d].max())) + 1 for d in range(dims)
        )

        # create roi sufficiently large to feed transformation
        source_roi = Roi(bb_min, bb_max - bb_min)

        return source_roi
