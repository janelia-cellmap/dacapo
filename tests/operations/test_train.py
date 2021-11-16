from dacapo.store.create_store import create_stats_store
from ..fixtures.runs import RUNS
from ..fixtures.db import options

from dacapo.experiments import RunConfig
from dacapo.compute_context import LocalTorch
from dacapo.store import create_config_store
from dacapo import train

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
    compute_context = LocalTorch(device="cpu")

    datasplit_config = datasplit_mkfunction(Path(options.runs_base_dir) / "data")
    run_config = RunConfig(
        name="test_run",
        task_config=task_config,
        architecture_config=architecture_config,
        trainer_config=trainer_config,
        datasplit_config=datasplit_config,
        repetition=0,
        num_iterations=100,
        snapshot_interval=5,
        validation_score="frizz_level",
        validation_score_minimize=False,
    )

    # create a store

    store = create_config_store()
    stats_store = create_stats_store()

    # store the configs

    store.store_run_config(run_config)

    # -------------------------------------

    # train

    train("test_run", compute_context=compute_context)

    # assert train_stats and validation_scores are available

    training_stats = stats_store.retrieve_training_stats("test_run")

    assert training_stats.trained_until() == 100