from ..fixtures.architectures import ARCHITECTURE_CONFIGS
from ..fixtures.datasplits import DATASPLIT_MK_FUNCTIONS
from ..fixtures.tasks import TASK_CONFIGS
from ..fixtures.trainers import TRAINER_CONFIGS

from dacapo import Options
from dacapo.store import create_config_store

from funlib.geometry import Coordinate, Roi

import pytest

NUM_ITER = 10


@pytest.mark.parametrize("architecture_config", ARCHITECTURE_CONFIGS)
@pytest.mark.parametrize("datasplit_mk_function", DATASPLIT_MK_FUNCTIONS)
@pytest.mark.parametrize("task_config", TASK_CONFIGS)
@pytest.mark.parametrize("trainer_config", TRAINER_CONFIGS)
def test_trainer(
    tmp_path,
    architecture_config,
    datasplit_mk_function,
    task_config,
    trainer_config,
):
    # Initialize the config store
    Options.instance(type="files", runs_base_dir=f"{tmp_path}")
    store = create_config_store()

    # Test store/retrieve
    store.store_trainer_config(trainer_config)
    fetched_array_config = store.retrieve_trainer_config(trainer_config.name)
    assert fetched_array_config == trainer_config

    # Create Trainer from config
    trainer = trainer_config.trainer_type(trainer_config)

    # Build prerequisites:
    # Trainer can only train with data and a model:
    datasplit_config = datasplit_mk_function(tmp_path)
    architecture = architecture_config.architecture_type(architecture_config)
    datasplit = datasplit_config.datasplit_type(datasplit_config)
    task = task_config.task_type(task_config)

    model = task.predictor.create_model(architecture)
    optimizer = trainer.create_optimizer(model)

    # Trainer must build its batch provider:
    trainer.build_batch_provider(datasplit.train, architecture, task)

    # enter the training context:
    with trainer as trainer:
        training_stats = list(trainer.iterate(NUM_ITER, model, optimizer))
        assert len(training_stats) == NUM_ITER
