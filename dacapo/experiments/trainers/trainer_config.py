import attr

from typing import Tuple


@attr.s
class TrainerConfig:
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
        return True, "No validation for this Trainer"
