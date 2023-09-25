import attr

from .affinities_task import AffinitiesTask
from .task_config import TaskConfig

from funlib.geometry import Coordinate

from typing import List


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
        default=False,
        metadata={
            "help_text": "Whether or not to train lsds along with your affinities. "
            "It has been shown that lsds as an auxiliary task can help affinity predictions."
        },
    )
    num_voxels: int = attr.ib(
        default=20,
        metadata={
            "help_text": "The number of voxels to use for the gaussian sigma when computing lsds."
        },
    )
    downsample_lsds: int = attr.ib(
        default=1,
        metadata={
            "help_text": "The amount to downsample the lsds. "
            "This is useful for speeding up training and inference."
        },
    )
    grow_boundary_iterations: int = attr.ib(
        default=0,
        metadata={
            "help_text": "The number of iterations to run the grow boundaries algorithm. "
            "This is useful for refining the boundaries of the affinities, and reducing merging of adjacent objects."
        },
    )
