from ..fixtures import *

from dacapo.experiments import Run
from dacapo.store.create_store import create_config_store, create_weights_store
from dacapo import predict

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
def test_predict(options, run_config, zarr_array, tmp_path):
    # os.environ["PYDEVD_UNBLOCK_THREADS_TIMEOUT"] = "2.0"

    # create a store

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
        run_config.name,
        iteration=0,
        input_container=zarr_array.file_name,
        input_dataset=zarr_array.dataset,
        output_path=tmp_path,
    )
    weights_store.store_weights(run, 1)
    predict(
        run_config.name,
        iteration=1,
        input_container=zarr_array.file_name,
        input_dataset=zarr_array.dataset,
        output_path=tmp_path,
    )

    # test predicting with iterations for which we know there are no weights
    with pytest.raises(ValueError):
        predict(
            run_config.name,
            iteration=2,
            input_container=zarr_array.file_name,
            input_dataset=zarr_array.dataset,
            output_path=tmp_path,
        )
