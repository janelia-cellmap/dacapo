import os
from upath import UPath as Path
import shutil
from ..fixtures import *

from dacapo.experiments import Run
from dacapo.store.create_store import create_config_store, create_weights_store
from dacapo import validate, validate_run

from dacapo.experiments.run_config import RunConfig

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
def test_validate(
    options,
    run_config,
):
    # set debug to True to run the test in a specific directory (for debugging)
    debug = False
    if debug:
        tmp_path = f"{Path(__file__).parent}/tmp"
        if os.path.exists(tmp_path):
            shutil.rmtree(tmp_path, ignore_errors=True)
        os.makedirs(tmp_path, exist_ok=True)
        old_path = os.getcwd()
        os.chdir(tmp_path)
    # when done debugging, delete "tests/operations/tmp"
    # -------------------------------------
    store = create_config_store()
    store.store_run_config(run_config)
    # validate
    validate(run_config.name, 0)
    # weights_store.store_weights(run, 1)
    # validate_run(run_config.name, 1)

    # test validating weights that don't exist
    with pytest.raises(FileNotFoundError):
        validate(run_config.name, 2)

    if debug:
        os.chdir(old_path)


@pytest.mark.parametrize(
    "run_config",
    [
        lf("distance_run"),
        lf("onehot_run"),
    ],
)
def test_validate_run(
    options,
    run_config,
):
    # set debug to True to run the test in a specific directory (for debugging)
    debug = False
    if debug:
        tmp_path = f"{Path(__file__).parent}/tmp"
        if os.path.exists(tmp_path):
            shutil.rmtree(tmp_path, ignore_errors=True)
        os.makedirs(tmp_path, exist_ok=True)
        old_path = os.getcwd()
        os.chdir(tmp_path)
    # when done debugging, delete "tests/operations/tmp"
    # -------------------------------------

    # create a store

    store = create_config_store()
    weights_store = create_weights_store()

    # store the configs

    store.store_run_config(run_config)

    run_config = store.retrieve_run_config(run_config.name)
    run = Run(run_config)

    # -------------------------------------

    # validate

    # test validating iterations for which we know there are weights
    weights_store.store_weights(run, 0)
    validate_run(run, 0)

    if debug:
        os.chdir(old_path)


@pytest.mark.parametrize("datasplit", [lf("six_class_datasplit")])
@pytest.mark.parametrize("task", [lf("distance_task"), lf("six_onehot_task")])
@pytest.mark.parametrize("trainer", [lf("gunpowder_trainer")])
@pytest.mark.parametrize("architecture", [lf("unet_architecture"), lf("unet_3d_architecture")])
def test_validate_unet(datasplit, task, trainer, architecture):
    store = create_config_store()
    weights_store = create_weights_store()

    run_config = RunConfig(
        name=f"{architecture.name}_run",
        task_config=task,
        architecture_config=architecture,
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
    weights_store.store_weights(run, 0)
    validate_run(run, 0)
