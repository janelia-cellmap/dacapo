from ..fixtures.runs import RUNS
from ..fixtures.db import options

from dacapo.experiments import RunConfig, Run
from dacapo.compute_context import LocalTorch
from dacapo.store import create_config_store, create_weights_store
from dacapo import validate

import pytest

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)


@pytest.mark.parametrize(
    "datasplit_mkfunction, architecture_config, task_config, trainer_config",
    RUNS,
)
def test_validate(
    options,
    datasplit_mkfunction,
    architecture_config,
    task_config,
    trainer_config,
):
    compute_context = LocalTorch(device="cpu")

    datasplit_config = datasplit_mkfunction(Path(options.runs_base_dir) / "data")
    run_config = RunConfig(
        name="test_run",
        task_config=task_config,
        architecture_config=architecture_config,
        trainer_config=trainer_config,
        datasplit_config=datasplit_config,
        repetition=0,
        num_iterations=3,
        validation_interval=1,
        snapshot_interval=5,
    )

    # create a store

    store = create_config_store()
    weights_store = create_weights_store()

    # store the configs

    store.store_run_config(run_config)

    run_config = store.retrieve_run_config("test_run")
    run = Run(run_config)

    # -------------------------------------

    # validate

    weights_store.store_weights(run, 0)
    validate(
        "test_run", 0, compute_context=compute_context
    )
    weights_store.store_weights(run, 1)
    validate(
        "test_run", 1, compute_context=compute_context
    )
