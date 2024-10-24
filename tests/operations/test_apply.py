import os
from upath import UPath as Path
import shutil
from ..fixtures import *

from dacapo.experiments import Run
from dacapo.store.create_store import create_config_store, create_weights_store
from dacapo import apply

import pytest
from pytest_lazy_fixtures import lf

import logging

logging.basicConfig(level=logging.INFO)


@pytest.mark.skip(reason="blockwise task is not currently supported")
@pytest.mark.parametrize(
    "run_config",
    [
        # lf("distance_run"),
        lf("dummy_run"),
        # lf("onehot_run"),
    ],
)
def test_apply(options, run_config, zarr_array, tmp_path):
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

    # apply
    parameters = list(run.task.post_processor.enumerate_parameters())[0]

    # test validating iterations for which we know there are weights
    weights_store.store_weights(run, 0)
    apply(
        run_config.name,
        zarr_array.file_name,
        zarr_array.dataset,
        output_path=tmp_path,
        iteration=0,
        parameters=parameters,
        num_workers=4,
    )
    weights_store.store_weights(run, 1)
    apply(
        run_config.name,
        zarr_array.file_name,
        zarr_array.dataset,
        output_path=tmp_path,
        iteration=1,
        parameters=parameters,
        num_workers=4,
    )

    # test validating weights that don't exist
    with pytest.raises(FileNotFoundError):
        apply(
            run_config.name,
            zarr_array.file_name,
            zarr_array.dataset,
            output_path=tmp_path,
            iteration=2,
            parameters=parameters,
            num_workers=4,
        )

    if debug:
        os.chdir(old_path)
