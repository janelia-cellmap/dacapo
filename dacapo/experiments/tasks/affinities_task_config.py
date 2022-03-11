import attr

from .affinities_task import AffinitiesTask
from .task_config import TaskConfig

from funlib.geometry import Coordinate

from typing import List, Tuple


@attr.s
class AffinitiesTaskConfig(TaskConfig):
    """This is a Affinities task config used for generating and
    evaluating voxel affinities for instance segmentations.
    """

    task_type = AffinitiesTask

    neighborhood: List[Coordinate] = attr.ib(
        metadata={
            "help_text": "The neighborhood upon which to calculate affinities. "
            "This is provided as a list of offsets, where each offset is a list of "
            "ints defining the offset in each axis in voxels."
        }
    )
    lsds: bool = attr.ib(
        metadata={
            "help_text": "Whether or not to train lsds along with your affinities. "
            "It has been shown that lsds as an auxiliary task can help affinity predictions."
        }
    )
