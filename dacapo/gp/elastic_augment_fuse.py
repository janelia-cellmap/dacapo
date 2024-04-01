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
    """
    Create an identity transformation grid.

    Args:
        shape (tuple): The shape of the transformation grid.
        voxel_size (tuple, optional): The voxel size of the grid. Defaults to None.
        offset (tuple, optional): The offset of the grid. Defaults to None.
        subsample (int, optional): The subsampling factor. Defaults to 1.
    Returns:
        numpy.ndarray: The identity transformation grid.
    Raises:
        AssertionError: If the subsample is not an integer.
    Examples:
        >>> _create_identity_transformation((10, 10, 10))
    """
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
    """
    Upscales a given transformation to match the specified output shape.

    Args:
        transformation (ndarray): The input transformation to be upscaled.
        output_shape (tuple): The desired output shape of the upscaled transformation.
        interpolate_order (int, optional): The order of interpolation used during upscaling. Defaults to 1.
        dtype (type, optional): The data type of the upscaled transformation. Defaults to np.float32.
    Returns:
        ndarray: The upscaled transformation with the specified output shape.
    Raises:
        AssertionError: If the transformation and output shape have different dimensions.
    Examples:
        >>> _upscale_transformation(transformation, (10, 10, 10))

    """

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
    """
    Rotate a point by a given angle.

    Args:
        point (list or tuple): The coordinates of the point to rotate.
        angle (float): The angle (in radians) by which to rotate the point.
    Returns:
        numpy.ndarray: The rotated point.
    Raises:
        AssertionError: If the point is not a list or tuple.
    Examples:
        >>> _rotate((1, 2), 0.5)

    """
    res = np.array(point)
    res[0] = math.sin(angle) * point[1] + math.cos(angle) * point[0]
    res[1] = -math.sin(angle) * point[0] + math.cos(angle) * point[1]

    return res


def _create_rotation_transformation(shape, angle, subsample=1, voxel_size=None):
    """
    Create a rotation transformation.

    Args:
        shape (tuple): The shape of the input volume.
        angle (float): The rotation angle in degrees.
        subsample (int, optional): The subsampling factor. Defaults to 1.
        voxel_size (tuple, optional): The voxel size of the input volume. Defaults to None.
    Returns:
        ndarray: The rotation transformation.
    Raises:
        AssertionError: If the subsample is not an integer.
    Examples:
        >>> _create_rotation_transformation((10, 10, 10), 0.5)
    Notes:
        The rotation is performed around the center of the volume.

    """
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
    """
    Create a uniform 3D transformation.

    Args:
        shape (tuple): The shape of the input volume.
        rotation (Rotation): The rotation to be applied to the control points.
        subsample (int, optional): The subsampling factor. Defaults to 1.
        voxel_size (Coordinate, optional): The voxel size of the input volume. Defaults to None.
    Returns:
        ndarray: The transformed control point offsets.
    Raises:
        AssertionError: If the subsample is not an integer.
    Examples:
        >>> _create_uniform_3d_transformation((10, 10, 10), Rotation.from_euler('xyz', [0.5, 0.5, 0.5]))
    Notes:
        The rotation is performed around the center of the volume.
    """
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
    """
    Calculate the minimum, maximum, mean, and standard deviation of the given ndarray.

    Args:
        ndarray (numpy.ndarray): The input ndarray.
    Returns:
        str: A string containing the calculated statistics with the given prefix.
    Raises:
        AssertionError: If the input is not a numpy array.
    Examples:
        >>> _min_max_mean_std(ndarray)
    """
    return ""


class ElasticAugment(BatchFilter):
    """
    Elasticly deform a batch. Requests larger batches upstream to avoid data
    loss due to rotation and jitter.
    Args:
        control_point_spacing (tuple of int): Distance between control points for the elastic deformation, in voxels per dimension.
        control_point_displacement_sigma (tuple of float): Standard deviation of control point displacement distribution, in world coordinates.
        rotation_interval (tuple of two float): Interval to randomly sample rotation angles from (0, 2PI).
        subsample (int): Instead of creating an elastic transformation on the full resolution, create one sub-sampled by the given factor, and linearly interpolate to obtain the full resolution transformation. This can significantly speed up this node, at the expense of having visible piecewise linear deformations for large factors. Usually, a factor of 4 can safely be used without noticeable changes. However, the default is 1 (i.e., no sub-sampling).
        seed (int): Set random state for reproducible results (tests only, do not use in production code!!)
        augmentation_probability (float): Probability to apply the augmentation.
        uniform_3d_rotation (bool): Use a uniform 3D rotation instead of a rotation around a random axis.
    Provides:
        * The arrays in the batch, deformed.
    Requests:
        * The arrays in the batch, enlarged such that the deformed ROI fits into
          the enlarged input ROI.
    Method:
        setup: Set up the ElasticAugment node.
        prepare: Prepare the ElasticAugment node.
        process: Process the ElasticAugment node.
    Notes:
        This node is a port of the ElasticAugment node from the original
        `gunpowder <
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
        """
        Initialize the BatchFilter object.

        Args:
            control_point_spacing (float): The spacing between control points.
            control_point_displacement_sigma (float): The standard deviation of the control point displacements.
            rotation_interval (tuple): A tuple containing the start and end angles for rotation.
            subsample (int, optional): The subsampling factor. Defaults to 1.
            augmentation_probability (float, optional): The probability of applying augmentation. Defaults to 1.0.
            seed (int, optional): The seed value for random number generation. Defaults to None.
            uniform_3d_rotation (bool, optional): Whether to use uniform 3D rotation. Defaults to False.
        Raises:
            AssertionError: If the subsample is not an integer.
        Examples:
            >>> ElasticAugment(control_point_spacing, control_point_displacement_sigma, rotation_interval, subsample=1, augmentation_probability=1.0, seed=None, uniform_3d_rotation=False)

        """
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
        """
        Set up the object by calculating the voxel size and spatial dimensions.

        This method calculates the voxel size by finding the minimum value for each axis
        from the voxel sizes of all array specs. It then sets the `voxel_size` attribute
        of the object. The spatial dimensions are also set based on the dimensions of the
        voxel size.

        Raises:
            AssertionError: If the voxel size is not a Coordinate object.
        Examples:
            >>> setup()

        """
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
        """
        Prepares the request for augmentation.

        Args:
            request: The request object containing the data to be augmented.
        Raises:
            AssertionError: If the key in the request is not an ArrayKey or GraphKey.
        Examples:
            >>> prepare(request)
        Notes:
            This method prepares the request for augmentation by performing the following steps:
            1. Logs the preparation details, including the transformation voxel size.
            2. Calculates the master ROI based on the total ROI.
            3. Generates a uniform random sample and determines whether to perform augmentation based on the augmentation probability.
            4. If augmentation is not required, logs the decision and returns.
            5. Snaps the master ROI to the grid based on the voxel size and calculates the master transformation.
            6. Clears the existing transformations and target ROIs.
            7. Iterates over each key in the request and prepares it for augmentation.
            8. Updates the upstream request with the modified ROI.

        """
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
        """
        Process the ElasticAugment node.

        Args:
            batch: The batch object containing the data to be processed.
            request: The request object specifying the data to be processed.
        Raises:
            AssertionError: If the key in the request is not an ArrayKey or GraphKey.
        Examples:
            >>> process(batch, request)
        Notes:
            This method applies the transformation to the data in the batch and restores the original ROIs.

        """
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
        """
        Create a transformation matrix for augmenting the input data.

        Args:
            target_shape (tuple): The shape of the target data.
            offset (tuple): The offset of the target data.
        Returns:
            np.ndarray: The transformation matrix.
        Raises:
            AssertionError: If the subsample is not an integer.
        Examples:
            >>> _create_transformation((10, 10, 10), (0, 0, 0))

        """
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
        """
        Returns a spatial region of interest (ROI) based on the given ROI.

        Args:
            roi (Roi): The input ROI.
        Returns:
            Roi: The spatial ROI.
        Raises:
            AssertionError: If the ROI is not a Roi object.
        Examples:
            >>> _spatial_roi(roi)

        """
        return Roi(
            roi.get_begin()[-self.spatial_dims :], roi.get_shape()[-self.spatial_dims :]
        )

    def _affine(self, array, scale, offset, target_roi, dtype=np.float32, order=1):
        """
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
        If a two-dimensional array was supplied, the output pixel value at index o was determined from the input image at position
        np.dot(matrix, o) + offset. This behavior was inconsistent and error-prone. As of version 0.18.0, the interpretation of
        the matrix parameter is consistent, and the offset parameter is always added to the input pixel index vector.

        Args:
            array (ndarray): The input array.
            scale (float or ndarray): The scale factor(s).
            offset (float or ndarray): The offset.
            target_roi (Roi): The target region of interest.
            dtype (type, optional): The data type of the output array. Defaults to np.float32.
            order (int, optional): The order of interpolation. Defaults to 1.
        Returns:
            ndarray: The transformed array.
        Raises:
            AssertionError: If the scale is not a scalar or 1-D array.
        Examples:
            >>> _affine(array, scale, offset, target_roi)
        References:
            taken from the scipy 0.18.1 doc:
            https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.ndimage.affine_transform.html#scipy.ndimage.affine_transform

        """
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
        """
        Shift the given transformation.

        Args:
            shift (tuple): The shift to apply to the transformation.
            transformation (ndarray): The transformation to shift.
        Returns:
            ndarray: The shifted transformation.
        Raises:
            AssertionError: If the shift is not a tuple.
        Examples:
            >>> _shift_transformation(shift, transformation)

        """
        for d in range(transformation.shape[0]):
            transformation[d] += shift[d]

    def _get_source_roi(self, transformation):
        """
        Get the source region of interest (ROI) for the given transformation.

        Args:
            transformation (ndarray): The transformation.
        Returns:
            Roi: The source ROI.
        Raises:
            AssertionError: If the transformation is not an ndarray.
        Examples:
            >>> transformation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            >>> _get_source_roi(transformation)
            Roi(Coordinate([0, 0, 0]), Coordinate([2, 2, 2]))
        Notes:
            Create the source ROI sufficiently large to feed the transformation.
        """
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
