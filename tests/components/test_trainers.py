from ..fixtures.architectures import ARCHITECTURE_CONFIGS
from ..fixtures.datasplits import DATASPLIT_MK_FUNCTIONS
from ..fixtures.tasks import TASK_CONFIGS
from ..fixtures.trainers import TRAINER_CONFIGS

from dacapo import Options
from dacapo.store import create_config_store

from funlib.geometry import Coordinate, Roi

import pytest


@pytest.mark.parametrize("architecture_config", ARCHITECTURE_CONFIGS)
@pytest.mark.parametrize("datasplit_mk_function", DATASPLIT_MK_FUNCTIONS)
@pytest.mark.parametrize("task_config", TASK_CONFIGS)
@pytest.mark.parametrize("trainer_config", TRAINER_CONFIGS)
def test_arrays(
    tmp_path,
    architecture_config,
    datasplit_mk_function,
    task_config,
    trainer_config,
):

    datasplit_config = datasplit_mk_function(tmp_path)

    architecture = architecture_config.architecture_type(architecture_config)
    datasplit = datasplit_config.datasplit_type(datasplit_config)
    task = task_config.task_type(task_config)
    trainer = trainer_config.trainer_type(trainer_config)

    trainer.build_batch_provider(datasplit, architecture, task)
