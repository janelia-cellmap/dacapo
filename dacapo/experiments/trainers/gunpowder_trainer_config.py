import attr

from .gp_augments import AugmentConfig
from .gunpowder_trainer import GunpowderTrainer
from .trainer_config import TrainerConfig

from typing import Optional, List


@attr.s
class GunpowderTrainerConfig(TrainerConfig):
    """
    This class is used to configure a Gunpowder Trainer. It contains attributes related to trainer type,
    number of data fetchers, augmentations to apply, snapshot interval, minimum masked value, and a boolean
    value indicating whether to clip raw or not.

    Attributes:
        trainer_type (class): This is the type of the trainer which is set to GunpowderTrainer by default.
        num_data_fetchers (int): This is the number of CPU workers who will be dedicated to fetch and process the data.
        augments (List[AugmentConfig]): This is the list of augments to apply during the training.
        snapshot_interval (Optional[int]): This is the number of iterations after which a new snapshot should be saved.
        min_masked (Optional[float]): This is the minimum masked value.
        clip_raw (bool): This is a boolean value indicating if the raw data should be clipped to the size of the GT data or not.
    """

    trainer_type = GunpowderTrainer

    num_data_fetchers: int = attr.ib(
        default=5,
        metadata={
            "help_text": "The number of cpu workers who will focus on fetching/processing "
            "data so that it is ready when the gpu needs it."
        },
    )

    augments: List[AugmentConfig] = attr.ib(
        factory=lambda: list(),
        metadata={"help_text": "The augments to apply during training."},
    )
    snapshot_interval: Optional[int] = attr.ib(
        default=None,
        metadata={"help_text": "Number of iterations before saving a new snapshot."},
    )
    min_masked: Optional[float] = attr.ib(default=0.15)
    clip_raw: bool = attr.ib(default=False)
    gt_min_reject: Optional[float] = attr.ib(
        default=None,
        metadata={
            "help_text": "The value to use for the GT mask. If None, the value will be "
            "determined by the GT mask augment."
            "e.g. 0.15"
        },
    )
