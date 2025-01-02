import os
from upath import UPath as Path
import shutil
from ..fixtures import *

from dacapo.experiments import Run
from dacapo.store.create_store import create_config_store, create_weights_store
from dacapo.predict_local import predict
from dacapo.store.array_store import LocalArrayIdentifier
import pytest
from pytest_lazy_fixtures import lf

import logging

logging.basicConfig(level=logging.INFO)


@pytest.mark.parametrize(
    "run_config",
    [
        # lf("distance_run"),
        lf("dummy_run"),
        # lf("onehot_run"),
    ],
)
def test_predict(options, run_config, zarr_array, tmp_path):
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
    input_identifier = LocalArrayIdentifier(
        Path(zarr_array.file_name), zarr_array.dataset
    )
    tmp_output = LocalArrayIdentifier(Path(tmp_path) / "prediciton.zarr", "prediction")
    store = create_config_store()
    weights_store = create_weights_store()

    # store the configs

    store.store_run_config(run_config)

    run_config = store.retrieve_run_config(run_config.name)
    run = Run(run_config)

    # -------------------------------------

    # predict
    # test predicting with iterations for which we know there are weights
    weights_store.store_weights(run, 0)
    predict(
        run.model,
        input_identifier,
        tmp_output,
    )
    if debug:
        os.chdir(old_path)
