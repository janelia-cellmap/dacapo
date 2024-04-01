import attr

from .dummy_trainer import DummyTrainer
from .trainer_config import TrainerConfig

from typing import Tuple


@attr.s
class DummyTrainerConfig(TrainerConfig):
    """
    This is just a dummy trainer config used for testing. None of the
    attributes have any particular meaning. This is just to test the trainer
    and the trainer config.

    Attributes:
        mirror_augment (bool): A boolean value indicating whether to use mirror
            augmentation or not.
    Methods:
        verify(self) -> Tuple[bool, str]: This method verifies the DummyTrainerConfig object.

    """

    trainer_type = DummyTrainer

    mirror_augment: bool = attr.ib(metadata={"help_text": "Dummy attribute."})

    def verify(self) -> Tuple[bool, str]:
        """
        Verify the DummyTrainerConfig object.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean value indicating whether the DummyTrainerConfig object is valid
                and a string containing the reason why the object is invalid.
        Examples:
            >>> valid, reason = trainer_config.verify()
        """
        return False, "This is a DummyTrainerConfig and is never valid"
