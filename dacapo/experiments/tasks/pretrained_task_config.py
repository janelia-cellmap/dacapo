import attr

from .pretrained_task import PretrainedTask
from .task_config import TaskConfig

from upath import UPath as Path


@attr.s
class PretrainedTaskConfig(TaskConfig):
    """
    Configuration for a task that uses a pretrained model. The model is loaded from a file
    and the weights are loaded from a file.

    Attributes:
        sub_task_config (TaskConfig): The task to run starting with the provided pretrained weights.
        weights (Path): A checkpoint containing pretrained model weights.
    Methods:
        verify(self) -> Tuple[bool, str]: This method verifies the PretrainedTaskConfig object.
    Notes:
        This is a subclass of TaskConfig.
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
