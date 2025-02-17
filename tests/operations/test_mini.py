from ..fixtures import *
from .helpers import (
    build_test_train_config,
    build_test_data_config,
    build_test_task_config,
    build_test_architecture_config,
)

from dacapo.store.create_store import create_array_store
from dacapo.experiments import Run
from dacapo.train import train_run
from dacapo.validate import validate_run
from dacapo.compute_context import create_compute_context
import zarr

import pytest

from dacapo.experiments.run_config import RunConfig

import pytest


# TODO: Move unet parameters that don't affect interaction with other modules
# to a separate architcture test
@pytest.mark.parametrize("data_dims", [2, 3])
@pytest.mark.parametrize("channels", [True, False])
@pytest.mark.parametrize("task", ["distance", "onehot", "affs"])
@pytest.mark.parametrize("architecture_dims", [2, 3])
@pytest.mark.parametrize("upsample", [True, False])
@pytest.mark.parametrize("padding", ["valid", "same"])
@pytest.mark.parametrize("func", ["train", "validate"])
@pytest.mark.parametrize("multiprocessing", [True, False])
def test_mini(
    tmpdir,
    data_dims,
    channels,
    task,
    architecture_dims,
    upsample,
    padding,
    func,
    multiprocessing,
):
    # Invalid configurations:
    if data_dims == 2 and architecture_dims == 3:
        # cannot train a 3D model on 2D data
        # TODO: maybe check that an appropriate warning is raised somewhere
        return

    trainer_config = build_test_train_config()

    data_config = build_test_data_config(
        tmpdir,
        data_dims,
        channels,
        upsample,
        "instance" if task == "affs" else "semantic",
    )
    task_config = build_test_task_config(task, data_dims, architecture_dims)
    architecture_config = build_test_architecture_config(
        data_dims,
        architecture_dims,
        channels,
        upsample,
        padding,
    )

    run_config = RunConfig(
        name=f"test_{func}",
        task_config=task_config,
        architecture_config=architecture_config,
        trainer_config=trainer_config,
        datasplit_config=data_config,
        repetition=0,
        num_iterations=2,
        snapshot_interval=1,
        batch_size=2,
        num_workers=int(multiprocessing),
    )
    run = Run(run_config)
    compute_context = create_compute_context()
    device = compute_context.device
    run.model.to(device)

    if func == "train":
        train_run(run, validate=False, save_snapshots=True)
        array_store = create_array_store()
        snapshot_container = array_store.snapshot_container(run.name).container
        assert snapshot_container.exists()
        assert all(
            x in zarr.open(snapshot_container)
            for x in [
                "volumes/raw",
                "volumes/gt",
                "volumes/target",
                "volumes/weight",
                "volumes/prediction",
                "volumes/gradients",
                "volumes/mask",
            ]
        )
        iterations_list = zarr.open(snapshot_container)["volumes/raw"].attrs[
            "iterations"
        ]
        assert iterations_list == [0, 1], iterations_list
    elif func == "validate":
        validate_run(run, 1)
