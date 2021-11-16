import attr

from .distance_task import DistanceTask
from .task_config import TaskConfig

from typing import List, Tuple


@attr.s
class DistanceTaskConfig(TaskConfig):
    """This is a Distance task config used for generating and
    evaluating signed distance transforms as a way of generating
    segmentations.

    The advantage of generating distance transforms over regular
    affinities is you can get a denser signal, i.e. 1 misclassified
    pixel in an affinity prediction could merge 2 otherwise very
    distinct objects, this cannot happen with distances.
    """

    task_type = DistanceTask

    channels: List[Tuple[str, List[int]]] = attr.ib(
        metadata={
            "help_text": "A list of channel names and id's in the gt that make up that channel."
        }
    )
    scale_factor: float = attr.ib(
        default=1,
        metadata={
            "help_text": "The amount by which to scale distances before applying "
            "a tanh normalization."
        },
    )
