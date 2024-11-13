from ..fixtures import *

import pytest
from pytest_lazy_fixtures import lf

import logging

logging.basicConfig(level=logging.INFO)


@pytest.mark.parametrize(
    "architecture_config",
    [
        lf("dummy_architecture"),
        lf("unet_architecture"),
    ],
)
def test_architecture(
    architecture_config,
):
    architecture = architecture_config.architecture_type(architecture_config)
    assert architecture.dims is not None, f"Architecture dims are None {architecture}"


@pytest.mark.parametrize(
    "architecture_config",
    [
        lf("dummy_architecture"),
        lf("unet_architecture"),
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

    architecture = retrieved_arch_config.architecture_type(retrieved_arch_config)

    assert architecture.dims is not None, f"Architecture dims are None {architecture}"
