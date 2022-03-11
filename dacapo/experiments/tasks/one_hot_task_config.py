import attr

from .one_hot_task import OneHotTask
from .task_config import TaskConfig

from typing import List


@attr.s
class OneHotTaskConfig(TaskConfig):
    """This is a One Hot prediction task that outputs a probability vector
    of length `c` for each voxel where `c` is the number of classes.
    Each voxel prediction has all positive values an l1 norm equal to 1.

    Post processing is extremely easy, the class of each voxel is
    simply the argmax over the vector of output probabilities.
    """

    task_type = OneHotTask

    classes: List[str] = attr.ib(
        metadata={"help_text": "The classes corresponding with each id starting from 0"}
    )
