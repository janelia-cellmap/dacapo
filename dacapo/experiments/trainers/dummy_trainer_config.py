import attr

from .dummy_trainer import DummyTrainer
from .trainer_config import TrainerConfig

from typing import Tuple


@attr.s
class DummyTrainerConfig(TrainerConfig):
    

    trainer_type = DummyTrainer

    mirror_augment: bool = attr.ib(metadata={"help_text": "Dummy attribute."})

    def verify(self) -> Tuple[bool, str]:
        
        return False, "This is a DummyTrainerConfig and is never valid"
