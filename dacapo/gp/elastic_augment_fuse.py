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
    Create an identity transformation with the specified parameters.

    Args:
        shape: tuple of ints, shape of the transformation.
        voxel_size: Coordinate object or None, size of a voxel.
        offset: Coordinate object or None, specifies the offset.
        subsample: Integer, specifies the subsampling factor.

    Returns:
        ndarray: multidimensional meshgrid with specified properties.
    """

    ...


def _upscale_transformation(
    transformation, output_shape, interpolate_order=1, dtype=np.float32
):
    """
    Rescale transformation to a new shape.

    Args:
        transformation: ndarray, input transformation.
        output_shape: tuple of ints, desired shape for the output transformation.
        interpolate_order: Integer, order of interpolation for resizing.
        dtype: dtype object, desired dtype for the output transformation.

    Returns:
        ndarray: Transformation of the desired shape.
    """
    ...
    
def _rotate(point, angle):
    """
    Rotate a point by a given angle.

    Args:
        point: ndarray, original coordinates of the point.
        angle: Float, angle in radians for the rotation.

    Returns:
        ndarray: Coordinates of the rotated point.
    """
    ...
    
def _create_rotation_transformation(shape, angle, subsample=1, voxel_size=None):
    """
    Create a rotation transformation for a given shape and angle.

    Args:
        shape: tuple of ints, shape of the transformation.
        angle: Float, angle in radians for the rotation.
        subsample: Integer, specifies the subsampling factor.
        voxel_size: Coordinate object or None, size of a voxel.

    Returns:
        ndarray: Rotation transformation.
    """
    ...

def _create_uniform_3d_transformation(shape, rotation, subsample=1, voxel_size=None):
    """
    Create a uniform 3D rotation transformation for a given shape and rotation matrix.

    Args:
        shape: tuple of ints, shape of the transformation.
        rotation: scipy.spatial.transform.Rotation object, specifies the rotation.
        subsample: Integer, specifies the subsampling factor.
        voxel_size: Coordinate object or None, size of a voxel.

    Returns:
        ndarray: Rotation transformation.
    """    
    ...

def _min_max_mean_std(ndarray, prefix=""):
    """
    Returns a string representation of the min, max, mean and standard deviation of an array.

    Args:
        ndarray: numpy array to calculate staticstics for.
        prefix: optional string that will be added in front of every statistics.

    Returns:
        String representation of the array statistics.
    """
    ...

class ElasticAugment(BatchFilter):
    """
    Elasticly deform a batch.

    Args:
        control_point_spacing (tuple of int): Distance between control points for the
            elastic deformation, in voxels per dimension.
        control_point_displacement_sigma (tuple of float):
            Standard deviation of control point displacement distribution, in world coordinates.
        rotation_interval (tuple of two floats): Interval to randomly sample rotation angles from (0, 2PI).
        subsample (int, optional): Instead of creating an elastic transformation on the full
            resolution, create one sub-sampled by the given factor, and linearly
            interpolate to obtain the full resolution transformation.
            Defaults to 1.
        augmentation_probability (float, optional): Value from 0 to 1 representing
            how often the augmentation will be applied.
            Defaults to 1.0.
        seed (int, optional): Set random state for reproducible results (tests only,
            do not use in production code!!). Defaults to None.
        uniform_3d_rotation (bool, optional): Whether to use 3D rotations. Defaults to False.
    """
    ...

    def prepare(self, request):
        """
        Prepare the batch filter for a given request.

        Args:
            request: The specifications of data for processing.
        """
        ...

    def process(self, batch, request):
        """
        Process the augmented batch.

        Args:
            batch: The actual batch to process.
            request: The specifications of data to process.
        """
        ...

    def _create_transformation(self, target_shape, offset):
        """
        Create a displacement transformation.

        Args:
            target_shape: tuple of ints, shape of the displacement.
            offset: offset for the displacement.
    
        Returns:
            ndarray: the displacement transformation.
        """
        ...

    def _spatial_roi(self, roi):
        """
        Get a spatial region of interest.

        Args:
            roi: The original region of interest.

        Returns:
            Roi: A new spatial region of interest.
        """
        ...

    def _affine(self, array, scale, offset, target_roi, dtype=np.float32, order=1):
        """
        Apply an affine transformation on an array.

        Args:
            array (ndarray): Array to be transformed.
            scale (float or ndarray): Scale of the transformation.
            offset (Coordinate): Offset for the transformation.
            target_roi (Roi): Region of Interest for target.
            dtype (dtype, optional): Datatype for the transformation.
            order (int, optional): Interpolation order for the transformation.
        
        Returns:
            ndarray: Object of the transformation.
        """
        ...

    def _shift_transformation(self, shift, transformation):
        """
        Shift a transformation.

        Args:
            shift (Coordinate): Shift to apply on transformation.
            transformation (ndarray): Transformation to shift.
        """
        ...
    
    def _get_source_roi(self, transformation):
        """
        Get the source region of interest for a transformation.

        Args:
            transformation: ndarray, the transformation.

        Returns:
            Roi: the source region of interest.
        """
        ...