from dacapo.experiments.architectures import (
    DummyArchitectureConfig,
    CNNectomeUNetConfig,
)

import pytest


@pytest.fixture()
def dummy_architecture():
    yield DummyArchitectureConfig(
        name="dummy_architecture", num_in_channels=1, num_out_channels=12
    )


@pytest.fixture()
def unet_architecture():
    yield CNNectomeUNetConfig(
        name="tmp_unet_architecture",
        input_shape=(2, 132, 132),
        eval_shape_increase=(8, 32, 32),
        fmaps_in=2,
        num_fmaps=8,
        fmaps_out=8,
        fmap_inc_factor=2,
        downsample_factors=[(1, 4, 4), (1, 4, 4)],
        kernel_size_down=[[(1, 3, 3)] * 2] * 3,
        kernel_size_up=[[(1, 3, 3)] * 2] * 2,
        constant_upsample=True,
        padding="valid",
    )
