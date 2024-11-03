from .augment_config import AugmentConfig
from dacapo.gp.elastic_augment_fuse import ElasticAugment

import attr

from typing import List, Tuple


@attr.s
class ElasticAugmentConfig(AugmentConfig):
    

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

    def node(self, _raw_key=None, _gt_key=None, _mask_key=None):
        
        return ElasticAugment(
            control_point_spacing=self.control_point_spacing,
            control_point_displacement_sigma=self.control_point_displacement_sigma,
            rotation_interval=self.rotation_interval,
            subsample=self.subsample,
            uniform_3d_rotation=self.uniform_3d_rotation,
        )
