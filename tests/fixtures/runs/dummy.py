from ..architectures import dummy_architecture_config
from ..datasplits import mk_dummy_datasplit
from ..tasks import dummy_task_config
from ..trainers import dummy_trainer_config

dummy_run = (
    mk_dummy_datasplit,
    dummy_architecture_config,
    dummy_task_config,
    dummy_trainer_config,
)
