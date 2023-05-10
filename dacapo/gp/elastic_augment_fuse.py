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
    """
    Elasticly deform a batch. Requests larger batches upstream to avoid data
    loss due to rotation and jitter.

    Args:

        control_point_spacing (``tuple`` of ``int``):

            Distance between control points for the elastic deformation, in
            voxels per dimension.

        control_point_displacement_sigma (``tuple`` of ``float``):

            Standard deviation of control point displacement distribution, in world coordinates.

        rotation_interval (``tuple`` of two ``floats``):

            Interval to randomly sample rotation angles from (0, 2PI).

        subsample (``int``):

            Instead of creating an elastic transformation on the full
            resolution, create one sub-sampled by the given factor, and linearly
            interpolate to obtain the full resolution transformation. This can
            significantly speed up this node, at the expense of having visible
            piecewise linear deformations for large factors. Usually, a factor
            of 4 can safely be used without noticeable changes. However, the
            default is 1 (i.e., no sub-sampling).

        seed (``int``):

            Set random state for reproducible results (tests only, do not use
            in production code!!)
    """

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
            "initialized with parameters "
            "control_point_spacing=%s "
            "control_point_displacement_sigma=%s "
            "rotation_start=%f "
            "rotation_max_amount=%f "
            "subsample=%f "
            "seed=%d",
            self.control_point_spacing,
            self.control_point_displacement_sigma,
            self.rotation_start,
            self.rotation_max_amount,
            self.subsample,
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
            "%s preparing request %s with transformation voxel size %s",
            type(self).__name__,
            request,
            self.voxel_size,
        )

        total_roi = request.get_total_roi()
        master_roi = self._spatial_roi(total_roi)
        logger.debug("master roi is %s with voxel size %s", master_roi, self.voxel_size)

        uniform_random_sample = np.random.rand()
        logger.debug(
            "Prepare: Uniform random sample is %f, probability to augment is %f",
            uniform_random_sample,
            self.augmentation_probability,
        )
        self.do_augment = uniform_random_sample < self.augmentation_probability
        if not self.do_augment:
            logger.debug(
                "Prepare: Randomly not augmenting at all. (probabilty to augment: %f)",
                self.augmentation_probability,
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
            "Master transformation statistics: %s", _min_max_mean_std(master_transform)
        )

        for key, spec in request.items():

            assert isinstance(key, ArrayKey) or isinstance(
                key, GraphKey
            ), "Only ArrayKey/GraphKey supported but got %s in request" % type(key)

            logger.debug("key %s: preparing with spec %s", key, spec)

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

            logger.debug("key %s: scale %s and offset %s", key, scale, offset)

            # need to pass inverse transform, hence -offset
            transform = self._affine(master_transform, scale, offset, target_roi_voxels)
            logger.debug(
                "key %s: transformed transform statistics          %s",
                key,
                _min_max_mean_std(transform),
            )
            source_roi = self._get_source_roi(transform).snap_to_grid(voxel_size)
            logger.debug(
                "key %s: source roi (target roi) is %s (%s)",
                key,
                source_roi,
                target_roi,
            )
            self._shift_transformation(-target_roi.get_begin(), transform)
            logger.debug(
                "key %s: shifted transformed transform statistics: %s",
                key,
                _min_max_mean_std(transform),
            )
            for d, (vs, b1, b2) in enumerate(
                zip(voxel_size, target_roi.get_begin(), source_roi.get_begin())
            ):
                pixel_offset = (b1 - b2) / vs
                transform[d] = transform[d] / vs + pixel_offset
            logger.debug(
                "key %s: pixel-space transform statistics:         %s",
                key,
                _min_max_mean_std(transform),
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
                "Process: Randomly not augmenting at all. (probabilty to augment: %f)",
                self.augmentation_probability,
            )
            return

        for key, _ in request.items():
            if isinstance(key, GraphKey):
                # restore original ROIs
                logger.warning("GRAPHS NOT PROPERLY SUPPORTED!")
                batch[key].spec.roi = request[key].roi
                continue

            assert key in batch.arrays, "only arrays supported but got %s" % key
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
                "key %s: applying transform with statistics %s %s",
                key,
                tuple(map(np.mean, self.transformations[key])),
                tuple(map(np.std, self.transformations[key])),
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
            "creating displacement for shape %s, subsample %d",
            target_shape,
            self.subsample,
        )
        transformation = _create_identity_transformation(
            target_shape,
            subsample=self.subsample,
            voxel_size=self.voxel_size,
            offset=offset,
        )
        if np.any(np.asarray(self.control_point_displacement_sigma) > 0):
            logger.debug(
                "Jittering with sigma=%s and spacing=%s",
                self.control_point_displacement_sigma,
                self.control_point_spacing,
            )
            elastic = augment.create_elastic_transformation(
                target_shape,
                self.control_point_spacing,
                self.control_point_displacement_sigma,
                subsample=self.subsample,
            )
            logger.debug(
                "elastic displacements statistics: %s", _min_max_mean_std(elastic)
            )
            transformation += elastic
        if not self.uniform_3d_rotation:
            rotation = (
                np.random.random() * self.rotation_max_amount + self.rotation_start
            )
            if rotation != 0:
                logger.debug("rotating with rotation=%f", rotation)
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
                "transform statistics before upscale: %s",
                _min_max_mean_std(transformation),
            )
            transformation = _upscale_transformation(transformation, target_shape)
            logger.debug(
                "transform statistics after  upscale: %s",
                _min_max_mean_std(transformation),
            )

        return transformation

    def _spatial_roi(self, roi):
        return Roi(
            roi.get_begin()[-self.spatial_dims :], roi.get_shape()[-self.spatial_dims :]
        )

    def _affine(self, array, scale, offset, target_roi, dtype=np.float32, order=1):
        """taken from the scipy 0.18.1 doc:
        https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.ndimage.affine_transform.html#scipy.ndimage.affine_transform

        Apply an affine transformation.
        The given matrix and offset are used to find for each point in the output the corresponding coordinates in the input by
        an affine transformation. The value of the input at those coordinates is determined by spline interpolation of the
        requested order. Points outside the boundaries of the input are filled according to the given mode.

        Given an output image pixel index vector o, the pixel value is determined from the input image at position
        np.dot(matrix,o) + offset.

        A diagonal matrix can be specified by supplying a one-dimensional array-like to the matrix parameter, in which case a
        more efficient algorithm is applied.

        Changed in version 0.18.0: Previously, the exact interpretation of the affine transformation depended on whether the
        matrix was supplied as a one-dimensional or two-dimensional array. If a one-dimensional array was supplied to the matrix
        parameter, the output pixel value at index o was determined from the input image at position matrix * (o + offset).
        """
        ndim = array.shape[0]
        output = np.empty((ndim,) + target_roi.get_shape(), dtype=dtype)
        for d in range(ndim):
            scipy.ndimage.affine_transform(
                input=array[d],
                matrix=scale,
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
