import attr

from typing import Tuple


@attr.s
class TrainerConfig:
    """
    A class to represent the Trainer Configurations.

    It is the base class for trainer configurations. Each subclass of a `Trainer`
    should have a specific config class derived from `TrainerConfig`.

    Attributes:
        name (str): A unique name for this trainer.
        batch_size (int): The batch size to be used during training.
        learning_rate (float): The learning rate of the optimizer.
    Methods:
        verify() -> Tuple[bool, str]:
            Verify whether this TrainerConfig is valid or not.
    Note:
        The TrainerConfig class is an abstract class that cannot be instantiated directly. It is meant to be subclassed.
    """

    name: str = attr.ib(
        metadata={
            "help_text": "A unique name for this trainer. This will be saved so you "
            "and others can find and reuse this trainer. Keep it short "
            "and avoid special characters."
        }
    )

    batch_size: int = attr.ib(
        metadata={
            "help_text": "The batch size to be used during training. Larger batch "
            "sizes will consume more memory per training iteration."
        },
    )

    learning_rate: float = attr.ib(
        metadata={"help_text": "The learning rate of the optimizer."},
    )

    def verify(self) -> Tuple[bool, str]:
        """
        Verify whether this TrainerConfig is valid or not.
        A TrainerConfig is considered valid if it has a valid batch size and learning rate.

        Returns:
            tuple: A tuple containing a boolean indicating whether the
            TrainerConfig is valid and a message explaining why.
        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        Examples:
            >>> valid, message = trainer_config.verify()
            >>> valid
            True
            >>> message
            "No validation for this Trainer"
        Note:
            This method must be implemented by the subclass.
        """
        return True, "No validation for this Trainer"
