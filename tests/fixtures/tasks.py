from dacapo.experiments.tasks import (
    DistanceTaskConfig,
    DummyTaskConfig,
    OneHotTaskConfig,
    HotDistanceTaskConfig,
)
import pytest


@pytest.fixture()
def dummy_task():
    yield DummyTaskConfig(name="dummy_task", embedding_dims=12, detection_threshold=0.1)


@pytest.fixture()
def distance_task():
    yield DistanceTaskConfig(
        name="distance_task",
        channels=[
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
        ],
        clip_distance=5,
        tol_distance=10,
    )


@pytest.fixture()
def hot_distance_task():
    yield HotDistanceTaskConfig(
        name="hot_distance_task",
        channels=[
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
        ],
        clip_distance=5,
        tol_distance=10,
    )


@pytest.fixture()
def onehot_task():
    yield OneHotTaskConfig(
        name="one_hot_task",
        classes=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"],
        kernel_size=1,
    )


@pytest.fixture()
def six_onehot_task():
    yield OneHotTaskConfig(
        name="one_hot_task",
        classes=["a", "b", "c", "d", "e", "f"],
        kernel_size=1,
    )
