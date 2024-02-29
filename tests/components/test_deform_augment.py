import pytest
from gunpowder import DeformAugment
from dacapo.experiments.trainers.gp_augments import DeformAugmentConfig


def test_deform_augment():
    config = DeformAugmentConfig(
        control_point_spacing=[10, 10, 10],
        jitter_sigma=[1, 1, 1],
        scale_interval=(0.9, 1.1),
        rotate=True,
        subsample=1,
        spatial_dims=3,
        use_fast_points_transform=False,
        recompute_missing_points=True,
        transform_key=None,
        graph_raster_voxel_size=[1, 1, 1],
    )

    node = config.node()
    assert isinstance(node, DeformAugment)
