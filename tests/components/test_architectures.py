import numpy as np
from funlib.persistence import prepare_ds
from funlib.geometry import Coordinate

import torch

from dacapo.experiments.datasplits import SimpleDataSplitConfig
from dacapo.experiments.tasks import (
    DistanceTaskConfig,
    OneHotTaskConfig,
    AffinitiesTaskConfig,
)
from dacapo.experiments.architectures import (
    CNNectomeUNetConfig,
    WrappedArchitectureConfig,
    ArchitectureConfig,
)

from pathlib import Path

from dacapo.experiments import Run
from dacapo.train import train_run
from dacapo.validate import validate_run

import pytest
from pytest_lazy_fixtures import lf

from dacapo.experiments.run_config import RunConfig

import pytest


def build_test_architecture_config(
    data_dims: int,
    architecture_dims: int,
    channels: bool,
    batch_norm: bool,
    upsample: bool,
    use_attention: bool,
    padding: str,
    wrapped: bool,
) -> ArchitectureConfig:
    """
    Build the simplest architecture config given the parameters.
    """
    if data_dims == 2:
        input_shape = (32, 32)
        eval_shape_increase = (8, 8)
        downsample_factors = [(2, 2)]
        upsample_factors = [(2, 2)] * int(upsample)

        kernel_size_down = [[(3, 3)] * 2] * 2
        kernel_size_up = [[(3, 3)] * 2] * 1
        kernel_size_down = None  # the default should work
        kernel_size_up = None  # the default should work

    elif data_dims == 3 and architecture_dims == 2:
        input_shape = (1, 32, 32)
        eval_shape_increase = (15, 8, 8)
        downsample_factors = [(1, 2, 2)]

        # test data upsamples in all dimensions so we have
        # to here too
        upsample_factors = [(2, 2, 2)] * int(upsample)

        # we have to force the 3D kernels to be 2D
        kernel_size_down = [[(1, 3, 3)] * 2] * 2
        kernel_size_up = [[(1, 3, 3)] * 2] * 1

    elif data_dims == 3 and architecture_dims == 3:
        input_shape = (32, 32, 32)
        eval_shape_increase = (8, 8, 8)
        downsample_factors = [(2, 2, 2)]
        upsample_factors = [(2, 2, 2)] * int(upsample)

        kernel_size_down = [[(3, 3, 3)] * 2] * 2
        kernel_size_up = [[(3, 3, 3)] * 2] * 1
        kernel_size_down = None  # the default should work
        kernel_size_up = None  # the default should work

    cnnectom_unet_config = CNNectomeUNetConfig(
        name="test_cnnectome_unet",
        input_shape=input_shape,
        eval_shape_increase=eval_shape_increase,
        fmaps_in=1 + channels,
        num_fmaps=2,
        fmaps_out=2,
        fmap_inc_factor=2,
        downsample_factors=downsample_factors,
        kernel_size_down=kernel_size_down,
        kernel_size_up=kernel_size_up,
        constant_upsample=True,
        upsample_factors=upsample_factors,
        batch_norm=batch_norm,
        use_attention=use_attention,
        padding=padding,
    )

    if wrapped:
        return WrappedArchitectureConfig(
            name="test_wrapped",
            module=cnnectom_unet_config.module(),
            fmaps_in=1 + channels,
            fmaps_out=2,
            input_shape=input_shape,
            scale=Coordinate(upsample_factors[0]) if upsample else None,
        )
    else:
        return cnnectom_unet_config


# TODO: Move unet parameters that don't affect interaction with other modules
# to a separate architcture test
@pytest.mark.parametrize("data_dims", [2, 3])
@pytest.mark.parametrize("channels", [True, False])
@pytest.mark.parametrize("architecture_dims", [2, 3])
@pytest.mark.parametrize("upsample", [True, False])
@pytest.mark.parametrize("batch_norm", [True, False])
@pytest.mark.parametrize("use_attention", [True, False])
@pytest.mark.parametrize("padding", ["valid", "same"])
@pytest.mark.parametrize("wrapped", [True, False])
def test_architectures(
    data_dims,
    channels,
    architecture_dims,
    batch_norm,
    upsample,
    use_attention,
    padding,
    wrapped,
):
    architecture_config = build_test_architecture_config(
        data_dims,
        architecture_dims,
        channels,
        batch_norm,
        upsample,
        use_attention,
        padding,
        wrapped,
    )

    in_data = torch.rand(
        (*(1, architecture_config.num_in_channels), *architecture_config.input_shape)
    )
    out_data = architecture_config.module()(in_data)

    assert out_data.shape[1] == architecture_config.num_out_channels
