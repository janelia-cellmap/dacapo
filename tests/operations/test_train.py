import numpy as np
from dacapo.store.create_store import create_stats_store
from ..fixtures import *

from dacapo.experiments import Run
from dacapo.store.create_store import create_config_store, create_weights_store
from dacapo.train import train_run

import pytest
from pytest_lazy_fixtures import lf

from dacapo.experiments.run_config import RunConfig

import logging

logging.basicConfig(level=logging.INFO)

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
    ],
)
def test_train(
    options,
    run_config,
):
    # create a store

    store = create_config_store()
    stats_store = create_stats_store()
    weights_store = create_weights_store()

    # store the configs

    store.store_run_config(run_config)
    run = Run(run_config)

    # -------------------------------------

    # train

    weights_store.store_weights(run, 0)
    train_run(run)

    init_weights = weights_store.retrieve_weights(run.name, 0)
    final_weights = weights_store.retrieve_weights(run.name, run.train_until)

    for name, weight in init_weights.model.items():
        weight_diff = (weight - final_weights.model[name]).sum()
        assert abs(weight_diff) > np.finfo(weight_diff.numpy().dtype).eps, weight_diff

    # assert train_stats and validation_scores are available

    training_stats = stats_store.retrieve_training_stats(run_config.name)

    assert training_stats.trained_until() == run_config.num_iterations


@pytest.mark.parametrize("datasplit", [lf("six_class_datasplit")])
@pytest.mark.parametrize("task", [lf("distance_task")])
@pytest.mark.parametrize("trainer", [lf("gunpowder_trainer")])
@pytest.mark.parametrize("batch_norm", [ False])
@pytest.mark.parametrize("upsample", [False])
@pytest.mark.parametrize("use_attention", [ False])
@pytest.mark.parametrize("three_d", [ False])
def test_train_unet(
    datasplit, task, trainer, batch_norm, upsample, use_attention, three_d
):
    architecture_config = unet_architecture_builder(
        batch_norm, upsample, use_attention, three_d
    )

    run_config = RunConfig(
        name=f"{architecture_config.name}_run",
        task_config=task,
        architecture_config=architecture_config,
        trainer_config=trainer,
        datasplit_config=datasplit,
        repetition=0,
        num_iterations=2,
    )
    run = Run(run_config)
    train_run(run)


@pytest.mark.parametrize("datasplit", [lf("six_class_datasplit")])
@pytest.mark.parametrize("task", [lf("distance_task")])
@pytest.mark.parametrize("trainer", [lf("gunpowder_trainer")])
@pytest.mark.parametrize("batch_norm", [True, False])
@pytest.mark.parametrize("upsample", [False])
@pytest.mark.parametrize("use_attention", [True, False])
@pytest.mark.parametrize("three_d", [True, False])
def test_train_unet(
    datasplit, task, trainer, batch_norm, upsample, use_attention, three_d
):
    store = create_config_store()
    stats_store = create_stats_store()
    weights_store = create_weights_store()

    architecture_config = unet_architecture_builder(
        batch_norm, upsample, use_attention, three_d
    )

    run_config = RunConfig(
        name=f"{architecture_config.name}_run",
        task_config=task,
        architecture_config=architecture_config,
        trainer_config=trainer,
        datasplit_config=datasplit,
        repetition=0,
        num_iterations=2,
    )
    try:
        store.store_run_config(run_config)
    except Exception as e:
        store.delete_run_config(run_config.name)
        store.store_run_config(run_config)

    run = Run(run_config)

    # -------------------------------------

    # train

    weights_store.store_weights(run, 0)
    train_run(run)

    init_weights = weights_store.retrieve_weights(run.name, 0)
    final_weights = weights_store.retrieve_weights(run.name, run.train_until)

    for name, weight in init_weights.model.items():
        weight_diff = (weight - final_weights.model[name]).any()
        assert weight_diff != 0, "Weights did not change"

    # assert train_stats and validation_scores are available

    training_stats = stats_store.retrieve_training_stats(run_config.name)

    assert training_stats.trained_until() == run_config.num_iterations





@pytest.mark.parametrize("upsample_datasplit", [lf("upsample_six_class_datasplit")])
@pytest.mark.parametrize("task", [lf("distance_task")])
@pytest.mark.parametrize("trainer", [lf("gunpowder_trainer")])
@pytest.mark.parametrize("batch_norm", [True, False])
@pytest.mark.parametrize("upsample", [True])
@pytest.mark.parametrize("use_attention", [True, False])
@pytest.mark.parametrize("three_d", [True, False])
def test_upsample_train_unet(
    upsample_datasplit, task, trainer, batch_norm, upsample, use_attention, three_d
):
    store = create_config_store()
    stats_store = create_stats_store()
    weights_store = create_weights_store()

    architecture_config = unet_architecture_builder(
        batch_norm, upsample, use_attention, three_d
    )

    run_config = RunConfig(
        name=f"{architecture_config.name}_run",
        task_config=task,
        architecture_config=architecture_config,
        trainer_config=trainer,
        datasplit_config=upsample_datasplit,
        repetition=0,
        num_iterations=2,
    )
    try:
        store.store_run_config(run_config)
    except Exception as e:
        store.delete_run_config(run_config.name)
        store.store_run_config(run_config)

    run = Run(run_config)

    # -------------------------------------

    # train

    weights_store.store_weights(run, 0)
    train_run(run)

    init_weights = weights_store.retrieve_weights(run.name, 0)
    final_weights = weights_store.retrieve_weights(run.name, run.train_until)

    for name, weight in init_weights.model.items():
        weight_diff = (weight - final_weights.model[name]).any()
        assert weight_diff != 0, "Weights did not change"

    # assert train_stats and validation_scores are available

    training_stats = stats_store.retrieve_training_stats(run_config.name)

    assert training_stats.trained_until() == run_config.num_iterations
