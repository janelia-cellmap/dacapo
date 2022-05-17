from dacapo.store.create_store import create_stats_store
from ..fixtures import *

from dacapo.compute_context import LocalTorch
from dacapo.store import create_config_store
from dacapo import train

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
def test_train(
    options,
    run_config,
):
    compute_context = LocalTorch(device="cpu")

    # create a store

    store = create_config_store()
    stats_store = create_stats_store()

    # store the configs

    store.store_run_config(run_config)

    # -------------------------------------

    # train

    train(run_config.name, compute_context=compute_context)

    # assert train_stats and validation_scores are available

    training_stats = stats_store.retrieve_training_stats(run_config.name)

    assert training_stats.trained_until() == run_config.num_iterations
