from ..fixtures.trainers import TRAINER_CONFIGS
from ..fixtures.db import options

from dacapo import Options
from dacapo.store import create_config_store

import pytest


@pytest.mark.parametrize("trainer_config", TRAINER_CONFIGS)
def test_trainer(
    options,
    trainer_config,
):
    # Initialize the config store (uses options behind the scene)
    store = create_config_store()

    # Test store/retrieve
    store.store_trainer_config(trainer_config)
    fetched_array_config = store.retrieve_trainer_config(trainer_config.name)
    assert fetched_array_config == trainer_config

    # Create Trainer from config
    trainer = trainer_config.trainer_type(trainer_config)
