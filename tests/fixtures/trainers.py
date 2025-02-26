from dacapo.experiments.trainers import DummyTrainerConfig, GunpowderTrainerConfig

import pytest


@pytest.fixture()
def dummy_trainer():
    yield DummyTrainerConfig(
        name="dummy_trainer", dummy_attr=True
    )


@pytest.fixture()
def gunpowder_trainer():
    yield GunpowderTrainerConfig(
        name="default_gp_trainer",
    )
