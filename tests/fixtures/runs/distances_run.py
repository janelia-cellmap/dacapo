from ..architectures import dummy_architecture_config
from ..datasplits import mk_six_class_distance_datasplit
from ..tasks import distance_task_config
from ..trainers import gunpowder_trainer_config

distance_run = (
    mk_six_class_distance_datasplit,
    dummy_architecture_config,
    distance_task_config,
    gunpowder_trainer_config,
)
