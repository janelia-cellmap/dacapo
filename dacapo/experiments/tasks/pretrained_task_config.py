import attr

from .pretrained_task import PretrainedTask
from .task_config import TaskConfig

from pathlib import Path


@attr.s
class PretrainedTaskConfig(TaskConfig):
    """
    Configuration class for a task that starts with pretrained weights.

    Attributes:
        task_type (Task): The type of the task.
        sub_task_config (TaskConfig): The configuration for the sub-task to run.
        weights (Path): A checkpoint containing pretrained model weights.
    """
    task_type = PretrainedTask

    sub_task_config: TaskConfig = attr.ib(
        metadata={
            "help_text": "The task to run starting with the provided pretrained weights."
        }
    )
    weights: Path = attr.ib(
        metadata={"help_text": "A checkpoint containing pretrained model weights."}
    )
