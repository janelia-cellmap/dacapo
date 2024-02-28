import attr

from typing import Any, Tuple
import tasks

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
    params: Any = attr.ib(
        default=None, metadata={"help_text": "Any additional options for this task."}
    )
    name: str = attr.ib(
        default="{experiment}_task",
        metadata={
            "help_text": "A unique name for this task. This will be saved so you and "
            "others can find and reuse this task. Keep it short and avoid "
            "special characters."
        },
    )

    def __attrs_post_init__(self, **kwargs):
        task_dict = {
            "semantic": "OneHot",
            "instance": "Affinity",
        }
        self.params = getattr(tasks, f"{task_dict[self.target]}TaskParameters")(
            **kwargs
        )
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.task_class = getattr(tasks, f"{task_dict[self.target]}Task")

    def verify(self) -> Tuple[bool, str]:
        """
        Check whether this is a valid Task
        """
        return True, "No validation for this Task"
