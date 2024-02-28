import attr

from typing import Tuple

# TODO: make lookup dynamically for tasks


@attr.s
class TaskConfig:
    """Base class for task configurations. Each subclass of a `Task` should
    have a corresponding config class derived from `TaskConfig`.
    """

    target: str = attr.ib(
        metadata={
            "help_text": "The target of task. This will be used to determine which "
            "Task class to use when generating targets, and computing loss, etc.. Examples are `semantic` and `instance` for simple semantic and instance segmentation respectively."
        }
    )
    name: str = attr.ib(
        default="{experiment}_task",
        metadata={
            "help_text": "A unique name for this task. This will be saved so you and "
            "others can find and reuse this task. Keep it short and avoid "
            "special characters."
        },
    )
    params: any = attr.ib(
        default={}, metadata={"help_text": "Any additional options for this task."}
    )

    def __init__(self, target: str, **kwargs):
        self.target = target
        task_dict = {
            "semantic": "OneHot",
            "instance": "Affinity",
        }
        self.params = any
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.config = getattr(subtasks, f"{target.capitalize()}TaskConfig")(**kwargs)

    def verify(self) -> Tuple[bool, str]:
        """
        Check whether this is a valid Task
        """
        return True, "No validation for this Task"
