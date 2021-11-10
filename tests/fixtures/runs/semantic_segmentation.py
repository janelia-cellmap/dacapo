from ..architectures import dummy_architecture_config
from ..datasplits import mk_twelve_class_datasplit
from ..tasks import dummy_task_config
from ..trainers import gunpowder_trainer_config

semantic_run = (
    mk_twelve_class_datasplit,
    dummy_architecture_config,
    dummy_task_config,
    gunpowder_trainer_config,
)
