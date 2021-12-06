import attr

from .gunpowder_trainer import GunpowderTrainer
from .trainer_config import TrainerConfig

from typing import Optional, Dict, Any


@attr.s
class GunpowderTrainerConfig(TrainerConfig):

    trainer_type = GunpowderTrainer

    num_data_fetchers: int = attr.ib(
        default=5,
        metadata={
            "help_text": "The number of cpu workers who will focus on fetching/processing "
            "data so that it is ready when the gpu needs it."
        },
    )

    simple_augment: Optional[Dict[str, Any]] = attr.ib(default=None)
    elastic_augment: Optional[Dict[str, Any]] = attr.ib(default=None)
    intensity_augment: Optional[Dict[str, Any]] = attr.ib(default=None)
    gamma_augment: Optional[Dict[str, Any]] = attr.ib(default=None)
    intensity_scale_shift: Optional[Dict[str, Any]] = attr.ib(default=None)
