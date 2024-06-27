import attr

from .hot_distance_task import HotDistanceTask
from .task_config import TaskConfig

from typing import List


@attr.s
class HotDistanceTaskConfig(TaskConfig):
    """
    Class for generating TaskConfigs for the HotDistanceTask, which predicts one hot encodings of classes, as well as signed distance transforms of those classes.

    Attributes:
        task_type: A reference to the Hot Distance Task class.
        channels (List[str]): A list of channel names.
        clip_distance (float): Maximum distance to consider for false positive/negatives.
        tol_distance (float): Tolerance distance for counting false positives/negatives.
        scale_factor (float): The amount by which to scale distances before applying
                              a tanh normalization. Defaults to 1.
        mask_distances (bool): Whether or not to mask out regions where the true distance to
                               object boundary cannot be known. Defaults to False
    Methods:
        verify(self) -> Tuple[bool, str]: This method verifies the HotDistanceTaskConfig object.
    Note:
        Generating distance transforms over regular affinities provides you with a denser
        signal, i.e., one misclassified pixel in an affinity prediction can merge 2
        otherwise very distinct objects, a situation that cannot happen with distances.
    """

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
