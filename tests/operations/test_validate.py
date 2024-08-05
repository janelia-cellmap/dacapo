import os
from pathlib import Path
import shutil
from ..fixtures import *

from dacapo.experiments import Run
from dacapo.store.create_store import create_config_store, create_weights_store
from dacapo import validate

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
    validate(run.name, 0, num_workers=4)
    # weights_store.store_weights(run, 1)
    # validate(run_config.name, 1, num_workers=4)

    # test validating weights that don't exist
    with pytest.raises(FileNotFoundError):
        validate(run.name, 2, num_workers=4)

    if debug:
        os.chdir(old_path)
