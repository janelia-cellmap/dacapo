import attr

from .hot_distance_task import HotDistanceTask
from .task_config import TaskConfig

from typing import List


@attr.s
class HotDistanceTaskConfig(TaskConfig):
    task_type = HotDistanceTask

    channels: List[str] = attr.ib(metadata={"help_text": "A list of channel names."})
    clip_distance: float = attr.ib(
        metadata={
            "help_text": "Maximum distance to consider for false positive/negatives."
        },
    )
    tol_distance: float = attr.ib(
        metadata={
            "help_text": "Tolerance distance for counting false positives/negatives"
        },
    )
    scale_factor: float = attr.ib(
        default=1,
        metadata={
            "help_text": "The amount by which to scale distances before applying "
            "a tanh normalization."
        },
    )
    mask_distances: bool = attr.ib(
        default=False,
        metadata={
            "help_text": "Whether or not to mask out regions where the true distance to "
            "object boundary cannot be known. This is anywhere that the distance to crop boundary "
            "is less than the distance to object boundary."
        },
    )
