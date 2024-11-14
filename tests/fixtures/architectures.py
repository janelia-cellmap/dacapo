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
        fmaps_in=1,
        num_fmaps=8,
        fmaps_out=8,
        fmap_inc_factor=2,
        downsample_factors=[(1, 4, 4), (1, 4, 4)],
        kernel_size_down=[[(1, 3, 3)] * 2] * 3,
        kernel_size_up=[[(1, 3, 3)] * 2] * 2,
        constant_upsample=True,
        padding="valid",
    )


@pytest.fixture()
def unet_3d_architecture():
    yield CNNectomeUNetConfig(
        name="tmp_unet_3d_architecture",
        input_shape=(188, 188, 188),
        eval_shape_increase=(72, 72, 72),
        fmaps_in=1,
        num_fmaps=6,
        fmaps_out=6,
        fmap_inc_factor=2,
        downsample_factors=[(2, 2, 2), (2, 2, 2), (2, 2, 2)],
        constant_upsample=True,
    )


def unet_architecture_builder(batch_norm, upsample, use_attention, three_d):
    name = "3d_unet" if three_d else "2d_unet"
    name = f"{name}_bn" if batch_norm else name
    name = f"{name}_up" if upsample else name
    name = f"{name}_att" if use_attention else name

    if three_d:
        return CNNectomeUNetConfig(
            name=name,
            input_shape=(188, 188, 188),
            eval_shape_increase=(72, 72, 72),
            fmaps_in=1,
            num_fmaps=6,
            fmaps_out=6,
            fmap_inc_factor=2,
            downsample_factors=[(2, 2, 2), (2, 2, 2), (2, 2, 2)],
            constant_upsample=True,
            upsample_factors=[(2, 2, 2)] if upsample else [],
            batch_norm=batch_norm,
            use_attention=use_attention,
        )
    else:
        return CNNectomeUNetConfig(
            name=name,
            input_shape=(2, 132, 132),
            eval_shape_increase=(8, 32, 32),
            fmaps_in=1,
            num_fmaps=8,
            fmaps_out=8,
            fmap_inc_factor=2,
            downsample_factors=[(1, 4, 4), (1, 4, 4)],
            kernel_size_down=[[(1, 3, 3)] * 2] * 3,
            kernel_size_up=[[(1, 3, 3)] * 2] * 2,
            constant_upsample=True,
            padding="valid",
            batch_norm=batch_norm,
            use_attention=use_attention,
            upsample_factors=[(1, 2, 2)] if upsample else [],
        )
