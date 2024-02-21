import attr
from .distance_task import DistanceTask
from .task_config import TaskConfig
from typing import List


@attr.s
class DistanceTaskConfig(TaskConfig):
    """This is a configuration class for the distance tasks.

    The class is used for generating and evaluating signed distance transforms.
    The advantage of generating distance transforms instead of regular affinities
    is that the signal can be denser. Misclassification of a single pixel in an affinity
    prediction can merge two distinct objects, but this does not occur with distances.

    Attributes:
        task_type: A constant attribute assigned to the DistanceTask.
        channels (List[str]): A list containing channel names.
        clip_distance (float): Maximum distance value to consider for false positive/negative evaluations.
        tol_distance (float): Tolerance level of distance for counting false positives/negatives.
        scale_factor (float): The factor by which distances are scaled before normalizing.
            Default is 1.
        mask_distances (bool): If True, masks out the regions where the true
            distance to object boundary cannot be accurately known.
            Default is False.
        clipmin (float): The minimum value allowed for distance weights. Default is 0.05.
        clipmax (float): The maximum value allowed for distance weights. Default is 0.95.
    """

    task_type = DistanceTask

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
    clipmin: float = attr.ib(
        default=0.05,
        metadata={"help_text": "The minimum value for distance weights."},
    )
    clipmax: float = attr.ib(
        default=0.95,
        metadata={"help_text": "The maximum value for distance weights."},
    )
