from ..fixtures import *

from dacapo.experiments import Run
from dacapo.compute_context import LocalTorch
from dacapo.store.create_store import create_config_store, create_weights_store
from dacapo import apply

import pytest
from pytest_lazyfixture import lazy_fixture

import logging

logging.basicConfig(level=logging.INFO)


@pytest.mark.parametrize(
    "run_config",
    [
        lazy_fixture("distance_run"),
        lazy_fixture("dummy_run"),
        lazy_fixture("onehot_run"),
    ],
)
def test_apply(
    options,
    run_config,
):
    # TODO: test the apply function
    return  # remove this line to run the test
    compute_context = LocalTorch(device="cpu")

    # create a store

    store = create_config_store()
    weights_store = create_weights_store()

    # store the configs

    store.store_run_config(run_config)

    run_config = store.retrieve_run_config(run_config.name)
    run = Run(run_config)

    # -------------------------------------

    # apply

    # test validating iterations for which we know there are weights
    weights_store.store_weights(run, 0)
    apply(run_config.name, 0, compute_context=compute_context)
    weights_store.store_weights(run, 1)
    apply(run_config.name, 1, compute_context=compute_context)

    # test validating weights that don't exist
    with pytest.raises(FileNotFoundError):
        apply(run_config.name, 2, compute_context=compute_context)
