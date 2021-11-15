from dacapo.store.create_store import create_stats_store, create_weights_store
from ..fixtures.runs import RUNS
from ..fixtures.db import options

from dacapo.experiments import RunConfig, Run
from dacapo.store import create_config_store
from dacapo import validate

import pytest

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)


@pytest.mark.parametrize(
    "datasplit_mkfunction, architecture_config, task_config, trainer_config", RUNS
)
def test_train(
    options, datasplit_mkfunction, architecture_config, task_config, trainer_config
):


    datasplit_config = datasplit_mkfunction(Path(options.runs_base_dir) / "data")
    run_config = RunConfig(
        name="test_run",
        task_config=task_config,
        architecture_config=architecture_config,
        trainer_config=trainer_config,
        datasplit_config=datasplit_config,
        repetition=0,
        num_iterations=1,
        validation_interval=1,
        snapshot_interval=5,
        validation_score="frizz_level",
        validation_score_minimize=False,
    )

    # create a store

    store = create_config_store()
    weights_store = create_weights_store()

    # store the configs

    store.store_run_config(run_config)

    run_config = store.retrieve_run_config("test_run")
    run = Run(run_config)

    weights_store.store_weights(run, 0)

    # -------------------------------------

    # train

    best_parameters, best_scores = validate("test_run", 0)
