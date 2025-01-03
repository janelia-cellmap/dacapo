from ..fixtures import *

import pytest
from pytest_lazy_fixtures import lf
import torch.nn as nn
from dacapo.experiments import Run
import logging

logging.basicConfig(level=logging.INFO)


@pytest.mark.parametrize(
    "architecture_config",
    [
        lf("dummy_architecture"),
        lf("unet_architecture"),
        lf("unet_3d_architecture"),
    ],
)
def test_stored_architecture(
    architecture_config,
):
    from dacapo.store.create_store import create_config_store

    config_store = create_config_store()
    try:
        config_store.store_architecture_config(architecture_config)
    except:
        config_store.delete_architecture_config(architecture_config.name)
        config_store.store_architecture_config(architecture_config)

    retrieved_arch_config = config_store.retrieve_architecture_config(
        architecture_config.name
    )

    architecture = retrieved_arch_config

    assert (
        architecture.dims is not None
    ), f"Architecture dims are None {architecture}"


@pytest.mark.parametrize(
    "architecture_config",
    [
        lf("unet_3d_architecture"),
        lf("unet_architecture"),
    ],
)
def test_conv_dims(
    architecture_config,
):
    architecture = architecture_config.module()
    for name, module in architecture.named_modules():
        if isinstance(module, nn.Conv2d):
            raise ValueError(f"Conv2d found in 3d unet {name}")


@pytest.mark.parametrize(
    "run_config",
    [
        lf("unet_3d_distance_run"),
    ],
)
def test_3d_conv_unet_in_run(
    run_config,
):
    run = Run(run_config)
    model = run.model
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            raise ValueError(f"Conv2d found in 3d unet {name}")
