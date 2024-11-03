import attr

from .pretrained_task import PretrainedTask
from .task_config import TaskConfig

from upath import UPath as Path


@attr.s
class PretrainedTaskConfig(TaskConfig):
    

    task_type = PretrainedTask

    sub_task_config: TaskConfig = attr.ib(
        metadata={
            "help_text": "The task to run starting with the provided pretrained weights."
        }
    )
    weights: Path = attr.ib(
        metadata={"help_text": "A checkpoint containing pretrained model weights."}
    )
