from dacapo.experiments.tasks.evaluators import BinarySegmentationEvaluator

import pytest


@pytest.fixture()
def binary_3_channel_evaluator():
    yield BinarySegmentationEvaluator(
        clip_distance=5, tol_distance=10, channels=["a", "b", "c"]
    )
