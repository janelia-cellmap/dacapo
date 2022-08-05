from dacapo.experiments.tasks.predictors import DistancePredictor, OneHotPredictor

import pytest


@pytest.fixture()
def distance_predictor():
    yield DistancePredictor(channels=["a", "b", "c"], scale_factor=50)


@pytest.fixture()
def onehot_predictor():
    yield OneHotPredictor(classes=["a", "b", "c"])
