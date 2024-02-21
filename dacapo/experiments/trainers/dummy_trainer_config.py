import attr
from .dummy_trainer import DummyTrainer
from .trainer_config import TrainerConfig
from typing import Tuple


@attr.s
class DummyTrainerConfig(TrainerConfig):
    """
    A subclass of TrainerConfig representing a dummy trainer configuration
    used for testing.

    Attributes:
        trainer_type (DummyTrainer): An instance of the DummyTrainer class.
        mirror_augment (bool): A dummy attribute with no actual purpose.

    """

    trainer_type = DummyTrainer
    mirror_augment: bool = attr.ib(metadata={"help_text": "Dummy attribute."})

    def verify(self) -> Tuple[bool, str]:
        """
        Dummy method to verify the configuration.

        This method will always return False and an error message as this is
        not meant to represent a valid trainer configuration.

        Returns:
            Tuple[bool, str]: False and a string indicating that the configuration is invalid.
        """

        return False, "This is a DummyTrainerConfig and is never valid"
