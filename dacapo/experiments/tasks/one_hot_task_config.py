import attr

from .one_hot_task import OneHotTask
from .task_config import TaskConfig

from typing import List


@attr.s
class OneHotTaskConfig(TaskConfig):
    

    task_type = OneHotTask

    classes: List[str] = attr.ib(
        metadata={"help_text": "The classes corresponding with each id starting from 0"}
    )
