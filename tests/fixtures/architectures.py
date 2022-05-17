from dacapo.experiments.architectures import DummyArchitectureConfig

import pytest


@pytest.fixture()
def dummy_architecture():
    yield DummyArchitectureConfig(
        name="dummy_architecture", num_in_channels=1, num_out_channels=12
    )
