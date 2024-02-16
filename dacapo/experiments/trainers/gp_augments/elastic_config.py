from .augment_config import AugmentConfig
from dacapo.gp.elastic_augment_fuse import ElasticAugment
import gunpowder as gp

import attr

from typing import List, Tuple


@attr.s
class DeformAugmentConfig(AugmentConfig):
    control_point_spacing: List[int] = attr.ib(
        metadata={
            "help_text": (
                "Distance between control points for the elastic deformation, in "
                "voxels per dimension."
            )
        }
    )
    jitter_sigma: List[float] = attr.ib(
        metadata={
            "help_text": (
                "Standard deviation of control point jitter distribution, in physical units per dimension."
            )
        }
    )
    scale_interval: Tuple[float, float] = attr.ib(
        metadata={"help_text": ("Interval to randomly sample scale factors from.")}
    )
    rotate: bool = attr.ib(
        metadata={"help_text": ("Whether or not to perform rotations.")}
    )
    subsample: int = attr.ib(
        default=1,
        metadata={
            "help_text": "Perform the elastic augmentation on a grid downsampled by this factor."
        },
    )
    spatial_dims: int = attr.ib(
        default=3,
        metadata={"help_text": "Number of spatial dimensions."},
    )
    use_fast_points_transform: bool = attr.ib(
        default=False,
        metadata={
            "help_text": """
            By solving for all of your points simultaneously with the following 3 step procedure:
            1) Rasterize nodes into numpy array
            2) Apply elastic transform to array
            3) Read out nodes via center of mass of transformed points
            You can gain substantial speed up as opposed to calculating the
            elastic transform for each point individually. However this may
            lead to nodes being lost during the transform.
            """
        },
    )
    recompute_missing_points: bool = attr.ib(
        default=True,
        metadata={
            "help_text": """Whether or not to compute the elastic transform node wise for nodes
            that were lossed during the fast elastic transform process."""
        },
    )
    transform_key: gp.ArrayKey = attr.ib(
        default=None,
        metadata={"help_text": "The key of the array to apply the transform to."},
    )
    graph_raster_voxel_size: List[float] = attr.ib(
        default=None,
        metadata={
            "help_text": "Voxel size of the rasterized graph. If None, the voxel size of the raw data is used."
        },
    )

    def node(self, _raw_key=None, _gt_key=None, _mask_key=None):
        return gp.DeformAugment(
            control_point_spacing=gp.Coordinate(self.control_point_spacing),
            jitter_sigma=gp.Coordinate(self.jitter_sigma),
            scale_interval=self.scale_interval,
            rotate=self.rotate,
            subsample=self.subsample,
            spatial_dims=self.spatial_dims,
            use_fast_points_transform=self.use_fast_points_transform,
            recompute_missing_points=self.recompute_missing_points,
            transform_key=self.transform_key,
            graph_raster_voxel_size=self.graph_raster_voxel_size,
        )


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
