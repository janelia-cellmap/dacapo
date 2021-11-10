from ..fixtures.runs import RUNS

from dacapo import Options
from dacapo.experiments import RunConfig
from dacapo.store import create_config_store
from dacapo import train

import logging

logging.basicConfig(level=logging.INFO)

import pytest


@pytest.mark.parametrize(
    "datasplit_mkfunction, architecture_config, task_config, trainer_config", RUNS
)
def test_train(
    tmp_path, datasplit_mkfunction, architecture_config, task_config, trainer_config
):  # create a run config
    Options._instance = None
    Options.instance(type="files", runs_base_dir=f"{tmp_path}")

    datasplit_config = datasplit_mkfunction(tmp_path)
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

    # store the configs

    store.store_run_config(run_config)

    # -------------------------------------

    # train

    train("test_run")
