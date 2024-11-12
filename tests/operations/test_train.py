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

from dacapo.experiments.architectures import (
    DummyArchitectureConfig,
    CNNectomeUNetConfig,
)

import pytest


def unet_architecture(batch_norm, upsample, use_attention, three_d):
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
            fmaps_in=2,
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
@pytest.mark.parametrize("batch_norm", [True, False])
@pytest.mark.parametrize("upsample", [True, False])
@pytest.mark.parametrize("use_attention", [True, False])
@pytest.mark.parametrize("three_d", [True, False])
def test_train_unet(
    datasplit, task, trainer, batch_norm, upsample, use_attention, three_d
):
    store = create_config_store()
    stats_store = create_stats_store()
    weights_store = create_weights_store()

    architecture_config = unet_architecture(
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
        weight_diff = (weight - final_weights.model[name]).sum()
        assert abs(weight_diff) > np.finfo(weight_diff.numpy().dtype).eps, weight_diff

    # assert train_stats and validation_scores are available

    training_stats = stats_store.retrieve_training_stats(run_config.name)

    assert training_stats.trained_until() == run_config.num_iterations
