from dacapo.experiments.run_config import RunConfig

import pytest


@pytest.fixture()
def distance_run(
    six_class_datasplit,
    dummy_architecture,
    distance_task,
    gunpowder_trainer,
):
    yield RunConfig(
        name="distance_run",
        task_config=distance_task,
        architecture_config=dummy_architecture,
        trainer_config=gunpowder_trainer,
        datasplit_config=six_class_datasplit,
        repetition=0,
        num_iterations=10,
    )


@pytest.fixture()
def hot_distance_run(
    six_class_datasplit,
    dummy_architecture,
    hot_distance_task,
    gunpowder_trainer,
):
    yield RunConfig(
        name="hot_distance_run",
        task_config=hot_distance_task,
        architecture_config=dummy_architecture,
        trainer_config=gunpowder_trainer,
        datasplit_config=six_class_datasplit,
        repetition=0,
        num_iterations=10,
    )


@pytest.fixture()
def dummy_run(
    dummy_datasplit,
    dummy_architecture,
    dummy_task,
    dummy_trainer,
):
    yield RunConfig(
        name="dummy_run",
        task_config=dummy_task,
        architecture_config=dummy_architecture,
        trainer_config=dummy_trainer,
        datasplit_config=dummy_datasplit,
        repetition=0,
        num_iterations=10,
    )


@pytest.fixture()
def onehot_run(
    twelve_class_datasplit,
    dummy_architecture,
    onehot_task,
    gunpowder_trainer,
):
    yield RunConfig(
        name="onehot_run",
        task_config=onehot_task,
        architecture_config=dummy_architecture,
        trainer_config=gunpowder_trainer,
        datasplit_config=twelve_class_datasplit,
        repetition=0,
        num_iterations=10,
    )


@pytest.fixture()
def weighted_onehot_run(
    twelve_class_datasplit,
    dummy_architecture,
    weighted_onehot_task,
    gunpowder_trainer,
):
    yield RunConfig(
        name="onehot_run",
        task_config=weighted_onehot_task,
        architecture_config=dummy_architecture,
        trainer_config=gunpowder_trainer,
        datasplit_config=twelve_class_datasplit,
        repetition=0,
        num_iterations=10,
    )


@pytest.fixture()
def unet_2d_distance_run(
    six_class_datasplit,
    unet_architecture,
    distance_task,
    gunpowder_trainer,
):
    yield RunConfig(
        name="unet_2d_distance_run",
        task_config=distance_task,
        architecture_config=unet_architecture,
        trainer_config=gunpowder_trainer,
        datasplit_config=six_class_datasplit,
        repetition=0,
        num_iterations=10,
    )


@pytest.fixture()
def unet_3d_distance_run(
    six_class_datasplit,
    unet_3d_architecture,
    distance_task,
    gunpowder_trainer,
):
    yield RunConfig(
        name="unet_3d_distance_run",
        task_config=distance_task,
        architecture_config=unet_3d_architecture,
        trainer_config=gunpowder_trainer,
        datasplit_config=six_class_datasplit,
        repetition=0,
        num_iterations=10,
    )
