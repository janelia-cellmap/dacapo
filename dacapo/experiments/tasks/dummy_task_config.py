import attr

from .dummy_task import DummyTask
from .task_config import TaskConfig

from typing import Tuple


@attr.s
class DummyTaskConfig(TaskConfig):
    """A class for creating a dummy task configuration object.

    This class extends the TaskConfig class and initializes dummy task configuration
    with default attributes. It is mainly used for testing aspects
    of the application without the need of creating real task configurations.

    Attributes:
        task_type (cls): The type of task. Here, set to DummyTask.
        embedding_dims (int): A dummy attribute represented as an integer.
        detection_threshold (float): Another dummy attribute represented as a float.
    Methods:
        verify(self) -> Tuple[bool, str]: This method verifies the DummyTaskConfig object.
    Note:
        This is a subclass of TaskConfig.

    """

    task_type = DummyTask

    embedding_dims: int = attr.ib(metadata={"help_text": "Dummy attribute."})

    detection_threshold: float = attr.ib(metadata={"help_text": "Dummy attribute."})

    def verify(self) -> Tuple[bool, str]:
        """A method to verify the dummy task configuration.

        Whenever called, this method always returns False and a statement showing
        that the DummyTaskConfig object is never valid.

        Returns:
            tuple: A tuple containing a boolean status and a string message.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> valid, reason = task_config.verify()
        """
        return False, "This is a DummyTaskConfig and is never valid"
