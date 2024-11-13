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

    architecture_type = architecture_config.architecture_type

    architecture = architecture_type(architecture_config)

    assert architecture.dims is not None, f"Architecture dims are None {architecture}"
