from ..fixtures import *

from dacapo.experiments import Run
from dacapo.store.create_store import create_config_store, create_weights_store
from dacapo import validate, validate_run

import pytest
from pytest_lazy_fixtures import lf

import logging

logging.basicConfig(level=logging.INFO)


@pytest.mark.parametrize(
    "run_config",
    [
        lf("distance_run"),
        lf("onehot_run"),
    ],
)
def test_large(
    options,
    run_config,
):
    store = create_config_store()
    weights_store = create_weights_store()
    store.store_run_config(run_config)

    # validate
    validate(run_config.name, 0)

    # validate_run
    run = Run(run_config)
    weights_store.store_weights(run, 1)
    validate_run(run, 1)

    # test validating weights that don't exist
    with pytest.raises(FileNotFoundError):
        validate(run_config.name, 2)

