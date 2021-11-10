import attr

from .gunpowder_trainer import GunpowderTrainer
from .trainer_config import TrainerConfig


@attr.s
class GunpowderTrainerConfig(TrainerConfig):

    trainer_type = GunpowderTrainer

    num_data_fetchers: int = attr.ib(
        default=5,
        metadata={
            "help_text": "The number of cpu workers who will focus on fetching/processing "
            "data so that it is ready when the gpu needs it."
        }
    )
