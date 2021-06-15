import attr

from .dummy_trainer import DummyTrainer
from .trainer_config import TrainerConfig


@attr.s
class DummyTrainerConfig(TrainerConfig):
    """This is just a dummy trainer config used for testing. None of the
    attributes have any particular meaning."""

    trainer_type = DummyTrainer

    mirror_augment: bool = attr.ib(
        metadata={
            "help_text":
                "Dummy attribute."
        }
    )
