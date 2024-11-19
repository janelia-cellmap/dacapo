from ..fixtures import *
from .helpers import (
    build_test_data_config,
    build_test_task_config,
    build_test_architecture_config,
)

from dacapo.experiments import Run
from dacapo.train import train_run
from dacapo.validate import validate_run

import pytest
from pytest_lazy_fixtures import lf

from dacapo.experiments.run_config import RunConfig

import pytest


# TODO: Move unet parameters that don't affect interaction with other modules
# to a separate architcture test
@pytest.mark.parametrize("data_dims", [2, 3])
@pytest.mark.parametrize("channels", [True, False])
@pytest.mark.parametrize("task", ["distance", "onehot", "affs"])
@pytest.mark.parametrize("trainer", [lf("gunpowder_trainer")])
@pytest.mark.parametrize("architecture_dims", [2, 3])
@pytest.mark.parametrize("upsample", [True, False])
# @pytest.mark.parametrize("batch_norm", [True, False])
@pytest.mark.parametrize("batch_norm", [False])
# @pytest.mark.parametrize("use_attention", [True, False])
@pytest.mark.parametrize("use_attention", [False])
@pytest.mark.parametrize("padding", ["valid", "same"])
@pytest.mark.parametrize("func", ["train", "validate"])
def test_mini(
    tmpdir,
    data_dims,
    channels,
    task,
    trainer,
    architecture_dims,
    batch_norm,
    upsample,
    use_attention,
    padding,
    func,
):
    # Invalid configurations:
    if data_dims == 2 and architecture_dims == 3:
        # cannot train a 3D model on 2D data
        # TODO: maybe check that an appropriate warning is raised somewhere
        return

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
        batch_norm,
        upsample,
        use_attention,
        padding,
    )

    run_config = RunConfig(
        name=f"test_{func}",
        task_config=task_config,
        architecture_config=architecture_config,
        trainer_config=trainer,
        datasplit_config=data_config,
        repetition=0,
        num_iterations=1,
    )
    run = Run(run_config)

    if func == "train":
        train_run(run)
    elif func == "validate":
        validate_run(run, 1)
