from dacapo.experiments.trainers import DummyTrainerConfig, GunpowderTrainerConfig

import pytest


@pytest.fixture()
def dummy_trainer():
    yield DummyTrainerConfig(
        name="dummy_trainer", learning_rate=1e-5, batch_size=10, mirror_augment=True
    )


@pytest.fixture()
def gunpowder_trainer():
    yield GunpowderTrainerConfig(
        name="default_gp_trainer",
        batch_size=1,
        learning_rate=0.0001,
        num_data_fetchers=1,
    )
