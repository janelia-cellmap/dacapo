import attr

from .inner_distance_task import InnerDistanceTask
from .task_config import TaskConfig

from typing import List


@attr.s
class InnerDistanceTaskConfig(TaskConfig):
    """A class to store configurations for inner distance tasks.

    This class inherits from TaskConfig to get configurations for signed distance
    transform tasks used for generating and evaluating segmentations. Compared to
    regular affinities, generating distance transforms can provide denser signals,
    avoiding situations like a single misclassified pixel merging two distinct objects.

    Attributes:
        task_type (InnerDistanceTask): The type of the task as InnerDistanceTask.
        channels (List[str]): A list holding names of channels.
        clip_distance (float): Maximum distance for considering false positives or negatives.
        tol_distance (float): Tolerance distance for counting false positives or negatives.
        scale_factor (float): The factor by which to scale distances before applying
                              a tanh normalization. Defaults to 1.
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
