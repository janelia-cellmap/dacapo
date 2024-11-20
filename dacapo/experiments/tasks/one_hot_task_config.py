import attr

from .one_hot_task import OneHotTask
from .task_config import TaskConfig

from typing import List


@attr.s
class OneHotTaskConfig(TaskConfig):
    """
    Class that derives from the TaskConfig to perform one hot prediction tasks.

    Attributes:
        task_type: the type of task, in this case, OneHotTask.
        classes: a List of classes which starts from id 0.

    Methods:
        None

    Note:
        The class of each voxel is simply the argmax over the vector of output probabilities.

    """

    task_type = OneHotTask

    classes: List[str] = attr.ib(
        metadata={"help_text": "The classes corresponding with each id starting from 0"}
    )
    kernel_size: int | None = attr.ib(
        default=None,
    )

    weights: list[float] = attr.ib(
        factory=list,
        metadata={
            "help_text": "Weights for the task. If not provided, the weights will be calculated based on representation "
        },
    )
