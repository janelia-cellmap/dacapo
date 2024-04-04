from ..fixtures import *

from dacapo.store.create_store import create_config_store

import pytest
from pytest_lazy_fixtures import lf


@pytest.mark.parametrize(
    "trainer_config",
    [
        lf("dummy_trainer"),
        lf("gunpowder_trainer"),
    ],
)
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
