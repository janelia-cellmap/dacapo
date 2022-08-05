from dacapo.experiments.tasks.post_processors import (
    ArgmaxPostProcessor,
    ThresholdPostProcessor,
)

import pytest


@pytest.fixture()
def argmax():
    yield ArgmaxPostProcessor()


@pytest.fixture()
def threshold():
    yield ThresholdPostProcessor()
