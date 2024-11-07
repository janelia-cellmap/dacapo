from .augment_config import AugmentConfig
from dacapo.gp.elastic_augment_fuse import ElasticAugment

import attr

from typing import List, Tuple


@attr.s
class ElasticAugmentConfig(AugmentConfig):
    """
    A class that holds the configuration details for the elastic augmentations.

    Attributes:
        control_point_spacing (List[int]): Distance(in voxels per dimension) between control points for
                                           the elastic deformation.
        control_point_displacement_sigma (List[float]): Standard deviation of control point displacement
                                                       distribution, in world coordinates.
        rotation_interval (Tuple[float, float]): An interval to randomly sample rotation angles from
                                                (0,2PI).
        subsample (int): Downsample factor to perform the elastic augmentation
                         on a grid. Default is 1.
        uniform_3d_rotation (bool): Should 3D rotations be performed uniformly. The 'rotation_interval'
                                    will be ignored. Default is False.
    Methods:
        node(_raw_key=None, _gt_key=None, _mask_key=None): Returns the object of ElasticAugment with the given
                                                          configuration details.

    """

    control_point_spacing: List[int] = attr.ib(
        metadata={
            "help_text": (
                "Distance between control points for the elastic deformation, in "
                "voxels per dimension."
            )
        }
    )
    control_point_displacement_sigma: List[float] = attr.ib(
        metadata={
            "help_text": (
                "Standard deviation of control point displacement distribution, in world coordinates."
            )
        }
    )
    rotation_interval: Tuple[float, float] = attr.ib(
        metadata={
            "help_text": ("Interval to randomly sample rotation angles from (0, 2PI).")
        }
    )
    subsample: int = attr.ib(
        default=1,
        metadata={
            "help_text": "Perform the elastic augmentation on a grid downsampled by this factor."
        },
    )
    uniform_3d_rotation: bool = attr.ib(
        default=False,
        metadata={
            "help_text": "Whether or not to perform rotations uniformly on a 3D sphere. This "
            "ignores the rotation interval due to the difficulty of parameterizing "
            "3D rotations."
        },
    )
    augmentation_probability: float = attr.ib(
        default=1.0,
        metadata={"help_text": "Probability of applying the augmentations."},
    )

    def node(self, _raw_key=None, _gt_key=None, _mask_key=None):
        """
        Returns the object of ElasticAugment with the given configuration details.

        Args:
            _raw_key: Unused variable, kept for future use.
            _gt_key: Unused variable, kept for future use.
            _mask_key: Unused variable, kept for future use.
        Returns:
            ElasticAugment: A ElasticAugment object configured with `control_point_spacing`,
                            `control_point_displacement_sigma`, `rotation_interval`, `subsample` and
                            `uniform_3d_rotation`.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> node = elastic_augment_config.node()

        """
        return ElasticAugment(
            control_point_spacing=self.control_point_spacing,
            control_point_displacement_sigma=self.control_point_displacement_sigma,
            rotation_interval=self.rotation_interval,
            subsample=self.subsample,
            uniform_3d_rotation=self.uniform_3d_rotation,
            augmentation_probability=self.augmentation_probability,
        )
