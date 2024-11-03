import attr

from .dummy_task import DummyTask
from .task_config import TaskConfig

from typing import Tuple


@attr.s
class DummyTaskConfig(TaskConfig):
    task_type = DummyTask

    embedding_dims: int = attr.ib(metadata={"help_text": "Dummy attribute."})

    detection_threshold: float = attr.ib(metadata={"help_text": "Dummy attribute."})

    def verify(self) -> Tuple[bool, str]:
        return False, "This is a DummyTaskConfig and is never valid"
