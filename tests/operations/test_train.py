import numpy as np
from dacapo.store.create_store import create_stats_store
from ..fixtures import *

from dacapo.experiments import RunConfig
from dacapo.store.create_store import create_config_store, create_weights_store
from dacapo.train import train_run

import pytest
from pytest_lazy_fixtures import lf

import pytest


# skip the test for the Apple Paravirtual device
# that does not support Metal 2.0
@pytest.mark.filterwarnings("ignore:.*Metal 2.0.*:UserWarning")
@pytest.mark.parametrize(
    "run_config",
    [
        lf("distance_run"),
        lf("dummy_run"),
        lf("onehot_run"),
        lf("hot_distance_run"),
    ],
)
def test_large(
    options,
    run_config: RunConfig,
):
    # create a store

    store = create_config_store()
    stats_store = create_stats_store()
    weights_store = create_weights_store()

    # store the configs

    store.store_run_config(run_config)
    run = run_config

    # -------------------------------------

    # train

    weights_store.store_weights(run, 0)
    train_run(run, validate=False)

    init_weights = weights_store.retrieve_weights(run.name, 0)
    final_weights = weights_store.retrieve_weights(run.name, run.train_until)

    weight_diffs = []
    for name, weight in init_weights.model.items():
        weight_diff = (weight - final_weights.model[name]).sum()
        weight_diffs.append((name, weight_diff))

    # test that some weights have changed instead of all weights have changed
    # depending on the design of the task and the model, some weights might
    # not get updated (dummy run for example)
    assert any(
        [
            abs(weight_diff) > np.finfo(weight_diff.numpy().dtype).eps
            for _, weight_diff in weight_diffs
        ]
    ), weight_diffs

    # assert train_stats and validation_scores are available

    training_stats = stats_store.retrieve_training_stats(run_config.name)

    assert training_stats.trained_until() == run_config.num_iterations
