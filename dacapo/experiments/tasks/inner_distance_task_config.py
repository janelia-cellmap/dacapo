import attr

from .inner_distance_task import InnerDistanceTask
from .task_config import TaskConfig

from typing import List


@attr.s
class InnerDistanceTaskConfig(TaskConfig):
    """
    This is a Distance task config used for generating and
    evaluating signed distance transforms as a way of generating
    segmentations.

    The advantage of generating distance transforms over regular
    affinities is you can get a denser signal, i.e. 1 misclassified
    pixel in an affinity prediction could merge 2 otherwise very
    distinct objects, this cannot happen with distances.

    Attributes:
        channels: A list of channel names.
        clip_distance: Maximum distance to consider for false positive/negatives.
        tol_distance: Tolerance distance for counting false positives/negatives
        scale_factor: The amount by which to scale distances before applying a tanh normalization.
    Notes:
        This is a subclass of TaskConfig.

    """

    task_type = InnerDistanceTask

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
