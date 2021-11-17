from ..architectures import dummy_architecture_config
from ..datasplits import mk_twelve_class_datasplit
from ..tasks import one_hot_task_config
from ..trainers import gunpowder_trainer_config

one_hot_run = (
    mk_twelve_class_datasplit,
    dummy_architecture_config,
    one_hot_task_config,
    gunpowder_trainer_config,
    "frizz_level",
    False,
)
