from dacapo.experiments.tasks.losses import DummyLoss

import pytest


@pytest.fixture()
def dummy_loss():
    yield DummyLoss()
